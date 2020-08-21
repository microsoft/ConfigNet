# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import time

from . import ConfigNetFirstStage, confignet_utils
from .metrics.metrics import ControllabilityMetrics
from .dnn_models.hologan_generator import HologanGenerator
from .dnn_models.real_encoder import RealEncoder
from .perceptual_loss import PerceptualLoss
from .confignet_first_stage import DEFAULT_CONFIG
from .losses import *


class ConfigNet(ConfigNetFirstStage):
    def __init__(self, config, initialize=True):
        self.config = confignet_utils.merge_configs(DEFAULT_CONFIG, config)

        super(ConfigNet, self).__init__(self.config, initialize=False)
        self.config["model_type"] = "ConfigNet"

        self.encoder = None
        self.generator_fine_tuned = None
        self.controllability_metrics = None
        self.perceptual_loss_face_reco = PerceptualLoss(self.config["output_shape"], model_type="VGGFace")

        if initialize:
            self.initialize_network()

    def get_weights(self):
        weights = super().get_weights()
        weights["real_encoder_weights"] = self.encoder.get_weights()

        return weights

    def set_weights(self, weights):
        super().set_weights(weights)
        self.encoder.set_weights(weights["real_encoder_weights"])

    def initialize_network(self):
        super(ConfigNet, self).initialize_network()

        self.encoder = RealEncoder(self.config["latent_dim"], self.config["output_shape"], self.config["rotation_ranges"])
        self.encoder(np.zeros((1, *self.config["output_shape"]), np.float32))

    # Start of checkpoint-related code
    def image_checkpoint(self, output_dir):
        self.synth_data_image_checkpoint(output_dir)

        # Autoencoder checkpoint start
        step_number = self.get_training_step_number()

        gt_imgs = self._checkpoint_visualization_input["input_images"]
        latent, pred_rotation = self.encode_images(gt_imgs)

        # Predicted latent with predicted rotation
        imgs_pred_rotations = self.generate_images(latent, pred_rotation)

        # Predicted latent with other rotations
        stacked_latents = np.vstack([latent] * self.n_checkpoint_rotations)
        imgs_rotation_sweep = self.generate_images(stacked_latents, self._checkpoint_visualization_input["rotation"])

        gt_imgs_0_255 = ((gt_imgs + 1) * 127.5).astype(np.uint8)
        combined_images = np.vstack((gt_imgs_0_255, imgs_pred_rotations, imgs_rotation_sweep))
        image_matrix = confignet_utils.build_image_matrix(combined_images, self.n_checkpoint_rotations + 2, self.n_checkpoint_samples)

        img_output_dir = os.path.join(output_dir, "output_imgs")
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)

        image_grid_file_path = os.path.join(img_output_dir, str(step_number).zfill(6) + ".png")
        cv2.imwrite(image_grid_file_path, image_matrix)
        with self.log_writer.as_default():
            tf.summary.image("generated_images", image_matrix[np.newaxis, :, :, [2, 1, 0]], step=step_number)

    def generate_output_for_metrics(self):
        latent, rotation = self.encode_images(self._generator_input_for_metrics["input_images"])
        return self.generate_images(latent, rotation)

    # End of checkpoint-related code

    # Start of training code
    def face_reco_loss(self, gt_imgs, gen_imgs):
        loss_vals = self.perceptual_loss_face_reco.loss(gen_imgs, gt_imgs)

        return tf.reduce_mean(loss_vals)

    def compute_normalized_latent_regression_loss(self, generator_outputs, labels):
        latent_regressor_output = self.latent_regressor(generator_outputs)

        denominator = tf.sqrt(tf.math.reduce_variance(labels, axis=0, keepdims=True) + 1e-3)
        # Do not normalize the rotation element
        denominator = tf.concat((denominator[:, :-3], tf.ones((1, 3), tf.float32)), axis=1)

        latent_regressor_output = tf.reduce_mean(latent_regressor_output, axis=0) + (latent_regressor_output - tf.reduce_mean(latent_regressor_output, axis=0)) / denominator
        labels = tf.reduce_mean(labels, axis=0) + (labels - tf.reduce_mean(labels, axis=0)) / denominator

        latent_regression_loss = tf.losses.mean_squared_error(labels, latent_regressor_output)
        latent_regression_loss = tf.reduce_mean(latent_regression_loss)
        latent_regression_loss *= self.config["latent_regression_weight"]

        return latent_regression_loss

    def sample_random_batch_of_images(self, dataset, batch_size=None):
        if batch_size is None:
            batch_size = self.get_batch_size()
        img_idxs = np.random.randint(0, dataset.imgs.shape[0], batch_size)
        imgs = np.copy(dataset.imgs[img_idxs])
        imgs = imgs.astype(np.float32) / 127.5 - 1.0
        imgs = confignet_utils.flip_random_subset_of_images(imgs)

        return imgs

    def get_discriminator_batch(self, training_set):
        # Inputs
        real_imgs = self.sample_random_batch_of_images(training_set)
        real_imgs = tf.convert_to_tensor(real_imgs)

        input_img_idxs = np.random.randint(0, training_set.imgs.shape[0], self.get_batch_size())
        input_imgs = training_set.imgs[input_img_idxs]
        input_imgs = input_imgs.astype(np.float32) / 127.5 - 1.0
        latent_vector, rotation = self.encode_images(input_imgs)
        fake_imgs = self.generator([latent_vector, rotation])

        return real_imgs, fake_imgs

    def latent_discriminator_training_step(self, real_training_set, synth_training_set, optimizer):
        # Inputs
        real_imgs = self.sample_random_batch_of_images(real_training_set)
        real_latents, _ = self.encoder(real_imgs)

        facemodel_params, _, _, _ = self.sample_synthetic_dataset(synth_training_set, self.get_batch_size())
        fake_latents = self.synthetic_encoder(facemodel_params)

        with tf.GradientTape() as tape:
            losses = compute_latent_discriminator_loss(self.latent_discriminator, real_latents, fake_latents)

        trainable_weights = self.latent_discriminator.trainable_weights
        gradients = tape.gradient(losses["loss_sum"], trainable_weights)
        optimizer.apply_gradients(zip(gradients, trainable_weights))

        return losses

    def generator_training_step(self, real_training_set, synth_training_set, optimizer):
        n_synth_in_batch = self.get_batch_size() // 2
        n_real_in_batch = self.get_batch_size() - n_synth_in_batch

        # Synth batch
        facemodel_params, synth_rotations, synth_imgs, eye_masks = self.sample_synthetic_dataset(synth_training_set, n_synth_in_batch)
        synth_imgs = synth_imgs.astype(np.float32) / 127.5 - 1.0

        # Real batch
        real_imgs = self.sample_random_batch_of_images(real_training_set, n_real_in_batch)

        # Labels for gan loss
        valid_y_synth = np.ones((n_synth_in_batch, 1))
        fake_y_real = np.zeros((n_real_in_batch, 1))

        domain_adverserial_loss_labels = np.vstack((fake_y_real, valid_y_synth))

        losses = {}
        with tf.GradientTape() as tape:
            synth_latents = self.synthetic_encoder(facemodel_params)

            generator_output_synth = self.generator((synth_latents, synth_rotations))

            real_latents, real_rotations = self.encoder(real_imgs)
            generator_output_real = self.generator((real_latents, real_rotations))

            losses["image_loss_synth"] = self.config["image_loss_weight"] * self.perceptual_loss.loss(synth_imgs, generator_output_synth)
            losses["image_loss_real"] = self.config["image_loss_weight"] * self.perceptual_loss.loss(real_imgs, generator_output_real)
            losses["eye_loss"] = self.config["eye_loss_weight"] * eye_loss(synth_imgs, generator_output_synth, eye_masks)

            # GAN loss for synth
            discriminator_output_synth = self.synth_discriminator(generator_output_synth)
            for i, disc_out in enumerate(discriminator_output_synth.values()):
                gan_loss = GAN_G_loss(disc_out)
                losses["GAN_loss_synth_" + str(i)] = gan_loss

            # GAN loss for real
            discriminator_output_real = self.discriminator(generator_output_real)
            for i, disc_out in enumerate(discriminator_output_real.values()):
                gan_loss = GAN_G_loss(disc_out)
                losses["GAN_loss_real_" + str(i)] = gan_loss

            # Domain adverserial loss
            latent_discriminator_out_synth = self.latent_discriminator(synth_latents)
            latent_discriminator_out_real = self.latent_discriminator(real_latents)

            latent_discriminator_output = tf.concat((latent_discriminator_out_real, latent_discriminator_out_synth), axis=0)

            latent_gan_loss = GAN_D_loss(domain_adverserial_loss_labels, latent_discriminator_output)

            losses["latent_GAN_loss"] = self.config["domain_adverserial_loss_weight"] * latent_gan_loss

            if self.config["latent_regression_weight"] > 0.0:
                # Latent regression loss start
                stacked_latent_vectors = tf.concat((synth_latents, real_latents), axis=0)
                stacked_generated_imgs = tf.concat((generator_output_synth, generator_output_real), axis=0)
                stacked_rotations = tf.concat((synth_rotations, real_rotations), axis=0)
                latent_regression_labels = tf.concat((stacked_latent_vectors, self.config["latent_regressor_rot_weight"] * stacked_rotations), axis=-1)

                # Regression of Z and rotation from output image
                losses["latent_regression_loss"] = self.compute_normalized_latent_regression_loss(stacked_generated_imgs, latent_regression_labels)

            losses["loss_sum"] = tf.reduce_sum(list(losses.values()))

        trainable_weights = self.generator.trainable_weights + self.latent_regressor.trainable_weights + self.synthetic_encoder.trainable_weights
        trainable_weights += self.encoder.trainable_weights
        gradients = tape.gradient(losses["loss_sum"], trainable_weights)
        optimizer.apply_gradients(zip(gradients, trainable_weights))

        return losses

    def calculate_metrics(self, output_dir, aml_run=None):
        super(ConfigNet, self).calculate_metrics(output_dir, aml_run)

        self.controllability_metrics.update_and_log_metrics(self._generator_input_for_metrics["input_images"],
                                                             self.metrics, output_dir, aml_run, self.log_writer)

        latents, rotations = self.encode_images(self._generator_input_for_metrics["input_images"])
        generator_input_dict = self.generator_smoothed.build_input_dict(latents, rotations)
        generated_images = self.generator_smoothed.predict(generator_input_dict)

        metric_batch_size = 16
        n_valid_samples = len(self._generator_input_for_metrics["input_images"])
        n_batches = 1 + n_valid_samples // metric_batch_size
        perceptual_loss = []
        for i in range(n_batches):
            start_idx = i * metric_batch_size
            end_idx = min(n_valid_samples, (i + 1) * metric_batch_size)
            gt_imgs = self._generator_input_for_metrics["input_images"][start_idx : end_idx]
            gen_imgs = generated_images[start_idx : end_idx]
            loss = self.perceptual_loss.loss(gt_imgs, gen_imgs)
            perceptual_loss.append(loss)

        perceptual_loss = float(np.mean(perceptual_loss))

        if "perceptual_loss" not in self.metrics.keys():
            self.metrics["perceptual_loss"] = []
        self.metrics["perceptual_loss"].append(perceptual_loss)

        if aml_run is not None:
            aml_run.log("perceptual_loss", perceptual_loss)
        with self.log_writer.as_default():
            tf.summary.scalar("metrics/perceptual_loss", perceptual_loss, step=self.get_training_step_number())

        np.savetxt(os.path.join(output_dir, "image_metrics.txt"), self.metrics["perceptual_loss"])

    def setup_training(self, log_dir, synth_training_set, n_samples_for_metrics, attribute_classifier, real_training_set, validation_set):
        super(ConfigNet, self).setup_training(log_dir, synth_training_set, n_samples_for_metrics, real_training_set)

        sample_idxs = np.random.randint(0, validation_set.imgs.shape[0], self.n_checkpoint_samples)
        checkpoint_input_imgs = validation_set.imgs[sample_idxs].astype(np.float32)
        self._checkpoint_visualization_input["input_images"] = (checkpoint_input_imgs / 127.5) - 1.0

        sample_idxs = np.random.randint(0, validation_set.imgs.shape[0], n_samples_for_metrics)
        metric_input_imgs = validation_set.imgs[sample_idxs].astype(np.float32)
        self._generator_input_for_metrics["input_images"] = (metric_input_imgs / 127.5) - 1.0

        self.controllability_metrics = ControllabilityMetrics(self, attribute_classifier)

    def train(self, real_training_set, synth_training_set, validation_set, attribute_classifier,
                output_dir, log_dir, n_steps=100000, n_samples_for_metrics=1000, aml_run=None):
        self.setup_training(log_dir, synth_training_set, n_samples_for_metrics, attribute_classifier,
                            real_training_set=real_training_set, validation_set=validation_set)
        start_step = self.get_training_step_number()

        discriminator_optimizer = keras.optimizers.Adam(**self.config["optimizer"])
        generator_optimizer = keras.optimizers.Adam(**self.config["optimizer"])

        for _ in range(start_step, n_steps):
            training_iteration_start = time.clock()

            for _ in range(self.config["n_discriminator_updates"]):
                d_loss = self.discriminator_training_step(real_training_set, discriminator_optimizer)
                synth_d_loss = self.synth_discriminator_training_step(synth_training_set, discriminator_optimizer)
                latent_d_loss = self.latent_discriminator_training_step(real_training_set, synth_training_set, discriminator_optimizer)

            for _ in range(self.config["n_generator_updates"]):
                g_loss = self.generator_training_step(real_training_set, synth_training_set, generator_optimizer)

            self.update_smoothed_weights()

            training_iteration_end = time.clock()
            print("[D loss: %f] [synth_D loss: %f] [latent_D_loss: %f] [G loss: %f]"%
                    (d_loss["loss_sum"], synth_d_loss["loss_sum"], latent_d_loss["loss_sum"], g_loss["loss_sum"]))
            confignet_utils.update_loss_dict(self.g_losses, g_loss)
            confignet_utils.update_loss_dict(self.d_losses, d_loss)
            confignet_utils.update_loss_dict(self.synth_d_losses, synth_d_loss)
            confignet_utils.update_loss_dict(self.latent_d_losses, latent_d_loss)

            iteration_time = training_iteration_end - training_iteration_start
            self.run_checkpoints(output_dir, iteration_time, aml_run=aml_run)

    def encode_images(self, input_images):
        if input_images.dtype == np.uint8:
            input_images = input_images.astype(np.float32)
            input_images = (input_images / 127.5) - 1.0

        embeddings, rotations = self.encoder.predict(input_images)

        return embeddings, rotations

    def generate_images(self, latent_vectors, rotations):
        generator_input_dict = self.generator.build_input_dict(latent_vectors, rotations)
        if self.generator_fine_tuned is not None:
            imgs = self.generator_fine_tuned.predict(generator_input_dict)
        else:
            imgs = self.generator_smoothed.predict(generator_input_dict)
        imgs = np.clip(imgs, -1.0, 1.0)
        imgs = ((imgs + 1) * 127.5).astype(np.uint8)

        return imgs

    def fine_tune_on_img(self, input_images, n_iters=50, img_output_dir=None, force_neutral_expression=False):
        if input_images.dtype == np.uint8:
            input_images = (input_images / 127.5) - 1.0
        if len(input_images.shape) == 3:
            input_images = input_images[np.newaxis]

        predicted_embeddings, predicted_rotations = self.encoder.predict(input_images)
        if force_neutral_expression:
            n_exp_blendshapes = self.config["facemodel_inputs"]["blendshape_values"][0]
            neutral_expr_params = np.zeros((1, n_exp_blendshapes), np.float32)
            predicted_embeddings = self.set_facemodel_param_in_latents(predicted_embeddings, "blendshape_values", neutral_expr_params)

        if self.generator_fine_tuned is None:
            self.generator_fine_tuned = HologanGenerator(**self._get_generator_kwargs())
            # Run once to generate weights
            self.generator_fine_tuned((predicted_embeddings, predicted_rotations))
        self.generator_fine_tuned.set_weights(self.generator_smoothed.get_weights())


        expr_idxs = self.get_facemodel_param_idxs_in_latent("blendshape_values")
        mean_predicted_embedding = np.mean(predicted_embeddings, axis=0, keepdims=True)

        pre_expr_embeddings = tf.Variable(mean_predicted_embedding[:, :expr_idxs[0]])
        expr_embeddings = tf.Variable(predicted_embeddings[:, expr_idxs])
        post_expr_embeddings = tf.Variable(mean_predicted_embedding[:, expr_idxs[-1] + 1:])
        n_imgs = input_images.shape[0]

        rotations = tf.Variable(predicted_rotations)

        optimizer = keras.optimizers.Adam(lr=0.0001)
        fake_y_real = np.ones((1, 1))

        convert_to_uint8 = lambda x: ((x[0] + 1) * 127.5).astype(np.uint8)

        if img_output_dir is not None:
            os.makedirs(img_output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(img_output_dir, "gt_img.png"), convert_to_uint8(input_images))

        for step_number in range(n_iters):
            losses = {}

            with tf.GradientTape() as tape:
                pre_expr_embeddings_tiled = tf.tile(pre_expr_embeddings, (n_imgs, 1))
                post_expr_embeddings_tiled = tf.tile(post_expr_embeddings, (n_imgs, 1))

                embeddings = tf.concat((pre_expr_embeddings_tiled, expr_embeddings, post_expr_embeddings_tiled), axis=1)

                generator_output_real = self.generator_fine_tuned((embeddings, rotations))
                losses["image_loss_real"] = 0.5 * self.config["image_loss_weight"] * self.perceptual_loss.loss(input_images, generator_output_real)
                losses["face_reco_loss"] = 0.5 * self.config["image_loss_weight"] * self.face_reco_loss(input_images, generator_output_real)

                # GAN loss for real
                discriminator_output_real = self.discriminator(generator_output_real)
                for i, disc_out in enumerate(discriminator_output_real.values()):
                    gan_loss = GAN_G_loss(disc_out)
                    losses["GAN_loss_real_" + str(i)] = gan_loss

                # Domain adverserial loss
                latent_discriminator_out_real = self.latent_discriminator(embeddings)
                latent_gan_loss = GAN_D_loss(fake_y_real, latent_discriminator_out_real)

                losses["latent_GAN_loss"] = self.config["domain_adverserial_loss_weight"] * latent_gan_loss

                # Latent regression loss start
                latent_regression_labels = tf.concat((embeddings, self.config["latent_regressor_rot_weight"] * rotations), axis=-1)

                # Regression of Z and rotation from output image
                losses["latent_regression_loss"] = self.compute_normalized_latent_regression_loss(generator_output_real, latent_regression_labels)

                losses["loss_sum"] = tf.reduce_sum(list(losses.values()))

            trainable_weights = self.generator_fine_tuned.trainable_weights + [pre_expr_embeddings, post_expr_embeddings, rotations]
            if not force_neutral_expression:
                trainable_weights.append(expr_embeddings)
            gradients = tape.gradient(losses["loss_sum"], trainable_weights)
            optimizer.apply_gradients(zip(gradients, trainable_weights))

            print(losses["loss_sum"])
            if img_output_dir is not None:
                cv2.imwrite(os.path.join(img_output_dir, "output_%02d.png"%(step_number)), convert_to_uint8(generator_output_real.numpy()))

        embeddings = tf.concat((pre_expr_embeddings_tiled, expr_embeddings, post_expr_embeddings_tiled), axis=1)
        return embeddings.numpy(), rotations.numpy()
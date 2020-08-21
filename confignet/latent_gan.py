# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json

from .confignet_utils import merge_configs
from .dnn_models.building_blocks import MLPSimple
from . import confignet_utils
from .metrics.metrics import InceptionMetrics
from .losses import GAN_D_loss, GAN_G_loss, gradient_regularization

DEFAULT_CONFIG = {
    "latent_dim": None,
    "optimizer": {
        "lr": 0.00005,
        "beta_1": 0.0,
        "beta_2": 0.9,
        "amsgrad": False
    },
    "batch_size": 32,
    "num_mlp_layers": 3,
    "latent_distribution_type": "normal",
    "hidden_layer_size_multiplier": 1.5,
    "n_samples_for_metrics": 1000,
    "verbose_log_period": 500,
    "logging_img_square_size": 6
}

class LatentGAN:
    def __init__(self, config):
        self.config = merge_configs(DEFAULT_CONFIG, config)

        self.generator = None
        self.generator_smoothed = None
        self.discriminator = None

        self.log_writer = None
        self.inputs_for_logs = None
        self.inputs_for_metrics = None

        self._inception_metric_object = None

        self.initialize_network()

    @classmethod
    def load(cls, file_path: str) -> "LatentGAN":
        with open(file_path, "r") as fp:
            config = json.load(fp)

        gan = cls(config)

        weight_file_path = os.path.splitext(file_path)[0] + ".npz"
        weights = np.load(weight_file_path, allow_pickle=True)
        gan.set_weights(weights)

        return gan

    def save(self, output_dir, output_filename):
        weights = self.get_weights()
        np.savez(os.path.join(output_dir, output_filename + ".npz"), **weights)
        with open(os.path.join(output_dir,output_filename + ".json"), "w") as fp:
            json.dump(self.config, fp, indent=4)

    def get_weights(self):
        weights = {}
        generator_weights = self.generator.get_weights()
        weights["generator_weights"] = np.empty(len(generator_weights), dtype=object)
        weights["generator_weights"][:] = generator_weights

        smoothed_generator_weights = self.generator_smoothed.get_weights()
        weights["smoothed_generator_weights"] = np.empty(len(smoothed_generator_weights), dtype=object)
        weights["smoothed_generator_weights"][:] = smoothed_generator_weights

        discriminator_weights = self.discriminator.get_weights()
        weights["discriminator_weights"] = np.empty(len(discriminator_weights), dtype=object)
        weights["discriminator_weights"][:] = discriminator_weights

        return weights

    def set_weights(self, weights):
        self.generator.set_weights(weights["generator_weights"])
        self.generator_smoothed.set_weights(weights["smoothed_generator_weights"])
        self.discriminator.set_weights(weights["discriminator_weights"])

    def initialize_network(self):
        self.generator = MLPSimple(num_layers=self.config["num_mlp_layers"],
                              num_in=self.config["latent_dim"],
                              num_hidden=int(self.config["latent_dim"] * self.config["hidden_layer_size_multiplier"]),
                              num_out=self.config["latent_dim"],
                              non_linear=keras.layers.LeakyReLU,
                              non_linear_last=None)

        self.generator_smoothed = MLPSimple(num_layers=self.config["num_mlp_layers"],
                              num_in=self.config["latent_dim"],
                              num_hidden=int(self.config["latent_dim"] * self.config["hidden_layer_size_multiplier"]),
                              num_out=self.config["latent_dim"],
                              non_linear=keras.layers.LeakyReLU,
                              non_linear_last=None)
        self.generator_smoothed.set_weights(self.generator.get_weights())

        self.discriminator = MLPSimple(num_layers=self.config["num_mlp_layers"],
                              num_in=self.config["latent_dim"],
                              num_hidden=int(self.config["latent_dim"] * self.config["hidden_layer_size_multiplier"]),
                              num_out=1,
                              non_linear=keras.layers.LeakyReLU,
                              non_linear_last=None)

    def sample_input_latent_vector(self, n_samples):
        if self.config["latent_distribution_type"] == "uniform":
            return np.random.uniform(-1, 1, (n_samples, self.config["latent_dim"]))
        elif self.config["latent_distribution_type"] == "normal":
            return np.random.normal(0, 1, (n_samples, self.config["latent_dim"]))

    def discriminator_training_step(self, gt_embeddings, optimizer):
        # Inputs
        latent_vectors = self.sample_input_latent_vector(self.config["batch_size"])
        fake_embeddings = self.generator.predict(latent_vectors)

        real_embedding_idxs = np.random.randint(0, gt_embeddings.shape[0], self.config["batch_size"])
        real_embeddings = gt_embeddings[real_embedding_idxs]
        real_embeddings = tf.convert_to_tensor(real_embeddings)

        # Labels
        valid_y = np.ones((real_embeddings.shape[0], 1))
        fake_y = np.zeros((fake_embeddings.shape[0], 1))

        losses = {}
        with tf.GradientTape() as tape:
            with tf.GradientTape() as grad_reg_tape:
                grad_reg_tape.watch(real_embeddings)
                discriminator_output_real = self.discriminator(real_embeddings)
            discriminator_output_fake = self.discriminator(fake_embeddings)

            # GAN loss on real
            losses["GAN_loss_real"] = GAN_D_loss(valid_y, discriminator_output_real)
            # GAN loss on fake
            losses["GAN_loss_fake"] = GAN_D_loss(fake_y, discriminator_output_fake)
            # Gradient penalty
            losses["gp_loss"] = gradient_regularization(grad_reg_tape, discriminator_output_real, real_embeddings)
            losses["loss_sum"] = tf.reduce_sum(list(losses.values()))

        trainable_weights = self.discriminator.trainable_weights
        gradients = tape.gradient(losses["loss_sum"], trainable_weights)
        optimizer.apply_gradients(zip(gradients, trainable_weights))

        return losses

    def generator_training_step(self, optimizer):
        latents = self.sample_input_latent_vector(self.config["batch_size"])

        losses = {}
        with tf.GradientTape() as tape:
            generated_embeddings = self.generator(latents)
            discriminator_outputs = self.discriminator(generated_embeddings)
            losses["gan_loss"] = GAN_G_loss(discriminator_outputs)
            losses["loss_sum"] = tf.reduce_sum(list(losses.values()))

        trainable_weights = self.generator.trainable_weights
        gradients = tape.gradient(losses["loss_sum"], trainable_weights)
        optimizer.apply_gradients(zip(gradients, trainable_weights))

        return losses

    def update_smoothed_weights(self, smoother_alpha=0.999):
        training_weights = self.generator.get_weights()
        smoothed_weights = self.generator_smoothed.get_weights()

        for i in range(len(smoothed_weights)):
            smoothed_weights[i] = smoother_alpha * smoothed_weights[i] + (1 - smoother_alpha) * training_weights[i]

        self.generator_smoothed.set_weights(smoothed_weights)

    def write_logs(self, output_dir, step_number, d_loss, g_loss, confignet_model):
        with self.log_writer.as_default():
            for key, value in d_loss.items():
                tf.summary.scalar("discr_" + key, value, step=step_number)
            for key, value in g_loss.items():
                tf.summary.scalar("gen_" + key, value, step=step_number)

        if step_number % self.config["verbose_log_period"] == 0:
            predicted_embeddings = self.generator_smoothed.predict(self.inputs_for_logs["latents"])
            generated_images = confignet_model.generate_images(predicted_embeddings, self.inputs_for_logs["rotations"])
            combined_image = confignet_utils.build_image_matrix(generated_images, self.config["logging_img_square_size"], self.config["logging_img_square_size"])
            combined_image = combined_image[:, :, [2, 1, 0]]
            with self.log_writer.as_default():
                tf.summary.image("generated_images", combined_image[np.newaxis], step=step_number)

            checkpoint_output_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(checkpoint_output_dir, exist_ok=True)
            self.save(checkpoint_output_dir, str(step_number).zfill(6))

            predicted_embeddings = self.generator_smoothed.predict(self.inputs_for_metrics["latents"])
            generated_images = confignet_model.generate_images(predicted_embeddings, self.inputs_for_metrics["rotations"])
            kid, fid = self._inception_metric_object.get_metrics(generated_images)
            with self.log_writer.as_default():
                tf.summary.scalar("metrics/kid", kid, step=step_number)
                tf.summary.scalar("metrics/fid", fid, step=step_number)

    def setup_logs(self, log_dir, training_set, confignet_model):
        os.makedirs(log_dir, exist_ok=True)
        self.log_writer = tf.summary.create_file_writer(log_dir)

        n_logged_images = self.config["logging_img_square_size"]**2
        self.inputs_for_logs = {}
        self.inputs_for_logs["latents"] = self.sample_input_latent_vector(n_logged_images)
        self.inputs_for_logs["rotations"] = np.zeros((n_logged_images, 3), np.float32)

        self._inception_metric_object = InceptionMetrics(confignet_model.config, training_set,
                                                         n_samples_for_metrics=self.config["n_samples_for_metrics"])

        self.inputs_for_metrics = {}
        self.inputs_for_metrics["latents"] = self.sample_input_latent_vector(self.config["n_samples_for_metrics"])
        self.inputs_for_metrics["rotations"] = confignet_model.sample_rotations(self.config["n_samples_for_metrics"])

    def extract_embeddings(self, confignet_model, training_set, max_chunk_size=1000):
        n_imgs = training_set.imgs.shape[0]
        embeddings = np.zeros((n_imgs, self.config["latent_dim"]), np.float32)

        n_chunks = 1 + n_imgs // max_chunk_size
        for i in range(n_chunks):
            print("Extracting embeddings for chunk %d of %d"%(i + 1, n_chunks))
            chunk_begin = i * max_chunk_size
            chunk_end = min((i + 1) * max_chunk_size, chunk_begin + n_imgs - chunk_begin)

            if chunk_end - chunk_begin == 0:
                break
            embeddings[chunk_begin : chunk_end], _ = confignet_model.encode_images(training_set.imgs[chunk_begin : chunk_end])

        return embeddings

    def train(self, training_set, confignet_model, output_dir, log_dir, n_iters):
        self.setup_logs(log_dir, training_set, confignet_model)

        gt_embeddings = self.extract_embeddings(confignet_model, training_set)
        optimizer = keras.optimizers.Adam(**self.config["optimizer"])

        for step_number in range(n_iters):
            d_loss = self.discriminator_training_step(gt_embeddings, optimizer)
            g_loss = self.generator_training_step(optimizer)

            self.update_smoothed_weights()

            print("[step: %d] [D loss: %f] [G loss: %f]"%(step_number, d_loss["loss_sum"], g_loss["loss_sum"]))
            self.write_logs(output_dir, step_number, d_loss, g_loss, confignet_model)

    def generate_latents(self, n_samples, truncation=1.0):
        input_latents = self.sample_input_latent_vector(n_samples) * truncation
        output_latents = self.generator_smoothed.predict(input_latents)

        return output_latents
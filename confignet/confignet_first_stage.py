# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
import pickle
import time
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
from collections import OrderedDict

from .dnn_models.synthetic_encoder import SyntheticDataEncoder
from .dnn_models.hologan_discriminator import HologanDiscriminator, HologanLatentRegressor
from .dnn_models.hologan_generator import HologanGenerator
from .dnn_models.building_blocks import MLPSimple
from .perceptual_loss import PerceptualLoss
from .losses import *
from .metrics.metrics import InceptionMetrics

from . import confignet_utils


DEFAULT_CONFIG = {
    "model_type": None,
    "latent_dim": 128,
    "output_shape": (128, 128, 3),
    "const_input_shape": (4, 4, 4, 512),
    "n_adain_mlp_layers": 2,
    "n_adain_mlp_units": 128,
    "gen_output_activation": "tanh",
    "n_discr_features_at_layer_0": 48,
    "max_discr_filters": 512,
    "n_discr_layers": 5,
    "discr_conv_kernel_size": 3,
    "latent_regression_weight": 10.0,
    "use_style_discriminator": True,
    "rotation_ranges": ((-30, 30), (-10, 10), (0, 0)),
    "relu_before_in": True,
    "initial_from_rgb_layer_in_discr": True,
    # True gets better metrics but reduces stability
    "adain_on_learned_input": False,
    # Increasing improves rotation stability
    "latent_regressor_rot_weight": 5.0,

    "optimizer": {
        "lr": 0.0004,
        "beta_1": 0.0,
        "beta_2": 0.9,
        "amsgrad": False
    },

    "batch_size": 24,
    "n_discriminator_updates": 1,
    "n_generator_updates": 1,
    "latent_distribution": "normal",
    "metrics_checkpoint_period": 1000,
    "image_checkpoint_period": 500,

    # Each input corresponds to a tuple where the first element is input dimensionality
    # the second element is corresponding dimensionality in latent space.
    # The first element should be filled by the process_metadata method of the training dataset.
    "facemodel_inputs": {
        "texture_embedding": (None, 30),
        "geometry_identity_params": (None, 30),
        "blendshape_values": (None, 30),
        "beard_style_embedding": (None, 7),
        "eyebrow_style_embedding": (None, 7),
        "lower_eyelash_style": (None, 2),
        "upper_eyelash_style": (None, 2),
        "head_hair_style_embedding": (None, 9),
        "eye_color": (None, 3),
        "head_hair_color": (None, 3),
        "hdri_embedding": (None, 20),
        "bone_rotations:left_eye": (None, 2)
    },

    "num_synth_encoder_layers": 2,
    "n_latent_discr_layers": 4,

    "image_loss_weight": 0.00005,
    "eye_loss_weight": 5,
    "domain_adverserial_loss_weight": 5.0
}

class ConfigNetFirstStage:
    def __init__(self, config, initialize=True):
        self.config = confignet_utils.merge_configs(DEFAULT_CONFIG, config)
        self.config["model_type"] = "ConfigNetFirstStage"

        self.generator = None
        self.generator_smoothed = None
        self.discriminator = None
        self.combined_model = None
        self.latent_regressor = None
        self.latent_discriminator = None
        self.synth_discriminator = None


        self.log_writer = None
        self.g_losses = {}
        self.d_losses = {}
        self.metrics = {}
        self.synth_d_losses = {}
        self.latent_d_losses = {}

        self._checkpoint_visualization_input = None
        self._generator_input_for_metrics = None
        self._inception_metric_object = None

        self.n_checkpoint_rotations = 6
        self.n_checkpoint_samples = 10

        # Remove the inputs that do not have a defined input dimensions
        self.config["facemodel_inputs"] = {key : value for key, value in self.config["facemodel_inputs"].items() if value[0] is not None}
        self.config["facemodel_inputs"] = OrderedDict(sorted(self.config["facemodel_inputs"].items(), key=lambda t: t[0]))

        self.config["latent_dim"] = 0
        for input_var_spec in self.config["facemodel_inputs"]:
            self.config["latent_dim"] += self.config["facemodel_inputs"][input_var_spec][1]

        self.synthetic_encoder = None
        self.facemodel_param_distributions = None
        self.perceptual_loss = PerceptualLoss(self.config["output_shape"], model_type="imagenet")

        if initialize:
            self.initialize_network()

    def get_weights(self, return_tensors=False):
        weights = {}
        weights["generator_weights"] = self.generator.get_weights()
        weights["generator_smoothed_weights"] = self.generator_smoothed.get_weights()
        weights["discriminator_weights"] = self.discriminator.get_weights()
        weights["latent_regressor_weights"] = self.latent_regressor.get_weights()
        weights["synthetic_encoder_weights"] = self.synthetic_encoder.get_weights()

        weights["latent_discriminator_weights"] = self.latent_discriminator.get_weights()
        weights["synth_discriminator_weights"] = self.synth_discriminator.get_weights()

        return weights

    def set_weights(self, weights):
        self.generator.set_weights(weights["generator_weights"])
        self.generator_smoothed.set_weights(weights["generator_smoothed_weights"])
        self.discriminator.set_weights(weights["discriminator_weights"])
        self.latent_regressor.set_weights(weights["latent_regressor_weights"])
        self.synthetic_encoder.set_weights(weights["synthetic_encoder_weights"])
        self.latent_discriminator.set_weights(weights["latent_discriminator_weights"])
        self.synth_discriminator.set_weights(weights["synth_discriminator_weights"])

    def get_training_step_number(self):
        step_number = 0 if "loss_sum" not in self.g_losses.keys() else len(self.g_losses["loss_sum"]) - 1

        return step_number

    def get_batch_size(self):
        return self.config["batch_size"]

    def get_log_dict(self):
        log_dict = {
            "g_losses": self.g_losses,
            "d_losses": self.d_losses,
            "metrics": self.metrics
        }

        return log_dict

    def set_logs(self, log_dict):
        self.g_losses = log_dict["g_losses"]
        self.d_losses = log_dict["d_losses"]
        self.metrics = log_dict["metrics"]

    def save(self, output_dir, output_filename):
        weights = self.get_weights()
        np.savez(os.path.join(output_dir, output_filename + ".npz"), **weights)
        with open(os.path.join(output_dir,output_filename + ".json"), "w") as fp:
            json.dump(self.config, fp, indent=4)

        with open(os.path.join(output_dir, output_filename + "_facemodel_distr.pck"), "wb") as fp:
            pickle.dump(self.facemodel_param_distributions, fp)

    @classmethod
    def load(cls, file_path):
        with open(file_path, "r") as fp:
            config = json.load(fp)

        model = cls(config)

        weigh_file = os.path.splitext(file_path)[0] + ".npz"
        weights = np.load(weigh_file, allow_pickle=True)
        model.set_weights(weights)

        log_file = os.path.splitext(file_path)[0] + "_log.json"
        if os.path.exists(log_file):
            with open(log_file, "r") as fp:
                log_dict = json.load(fp)
            model.set_logs(log_dict)

        path_to_distribution_file = os.path.splitext(file_path)[0] + "_facemodel_distr.pck"
        if os.path.exists(path_to_distribution_file):
            with open(path_to_distribution_file, "rb") as fp:
                model.facemodel_param_distributions = pickle.load(fp)
        else:
            print("WARNING: facemodel param distributions not loaded")

        return model

    # Returns the total number of facemodel input dimensions
    @property
    def facemodel_input_dim(self):
        total_facemodel_input_dims = 0
        for facemodel_input_dim, _ in self.config["facemodel_inputs"].values():
            total_facemodel_input_dims += facemodel_input_dim

        return total_facemodel_input_dims

    def get_facemodel_param_idxs_in_latent(self, param_name):
        facemodel_param_dims = list(self.config["facemodel_inputs"].values())
        facemodel_param_names = list(self.config["facemodel_inputs"].keys())

        facemodel_param_idx = facemodel_param_names.index(param_name)

        start_idx = int(np.sum([x[1] for x in facemodel_param_dims[:facemodel_param_idx]]))
        end_idx = start_idx + facemodel_param_dims[facemodel_param_idx][1]

        return range(start_idx, end_idx)

    def set_facemodel_param_in_latents(self, latents, param_name, param_value):
        param_value = np.array(param_value)
        if len(param_value.shape) == 1:
            param_value = param_value[np.newaxis]
        latents_for_param = self.synthetic_encoder.per_facemodel_input_mlps[param_name].predict(param_value)

        param_idxs_in_latent = self.get_facemodel_param_idxs_in_latent(param_name)

        new_latents = np.copy(latents)
        new_latents[:, param_idxs_in_latent] = latents_for_param

        return new_latents

    def _get_generator_kwargs(self):
        generator_kwargs = {
            "latent_dim": self.config["latent_dim"],
            "output_shape": self.config["output_shape"][:2],
            "n_adain_mlp_units": self.config["n_adain_mlp_units"],
            "n_adain_mlp_layers": self.config["n_adain_mlp_layers"],
            "gen_output_activation": self.config["gen_output_activation"]
        }
        return generator_kwargs

    def initialize_network(self):
        self.synthetic_encoder = SyntheticDataEncoder(synthetic_encoder_inputs=self.config["facemodel_inputs"],
                                                      num_layers=self.config["num_synth_encoder_layers"])
        discriminiator_args = {
            "img_shape": self.config["output_shape"][:2],
            "num_resample": self.config["n_discr_layers"],
            "disc_kernel_size": self.config["discr_conv_kernel_size"],
            "disc_expansion_factor": self.config["n_discr_features_at_layer_0"],
            "disc_max_feature_maps": self.config["max_discr_filters"],
            "initial_from_rgb_layer_in_discr": self.config["initial_from_rgb_layer_in_discr"]
        }
        discriminator_input_shape = tuple([self.config["batch_size"]] + list(self.config["output_shape"]))

        self.discriminator = HologanDiscriminator(**discriminiator_args)
        self.discriminator.build(discriminator_input_shape)
        self.synth_discriminator = HologanDiscriminator(**discriminiator_args)
        self.synth_discriminator.build(discriminator_input_shape)

        self.latent_discriminator = MLPSimple(num_layers=self.config["n_latent_discr_layers"],
                                              num_in=self.config["latent_dim"],
                                              num_hidden=self.config["latent_dim"],
                                              num_out=1,
                                              non_linear=keras.layers.LeakyReLU,
                                              non_linear_last=None)

        # Used in identity loss, described in supplementary
        self.latent_regressor = HologanLatentRegressor(self.config["latent_dim"], **discriminiator_args)
        self.latent_regressor.build(discriminator_input_shape)

        generator_kwargs = self._get_generator_kwargs()
        generator_input_shape = [(self.config["batch_size"], self.config["latent_dim"]), (self.config["batch_size"], 3)]
        self.generator = HologanGenerator(**generator_kwargs)
        self.generator.build(generator_input_shape)

        self.generator_smoothed = HologanGenerator(**generator_kwargs)
        self.generator_smoothed.build(generator_input_shape)
        self.generator_smoothed.set_weights(self.generator.get_weights())

    def synth_data_image_checkpoint(self, output_dir):
        step_number = self.get_training_step_number()

        facemodel_params = self._checkpoint_visualization_input["facemodel_params"]
        gt_imgs = self._checkpoint_visualization_input["gt_imgs"]
        rotation = self._checkpoint_visualization_input["rotation"]

        generated_imgs = self.generate_images_from_facemodel(facemodel_params, rotation)
        generated_imgs = np.vstack((gt_imgs, generated_imgs))

        combined_image = confignet_utils.build_image_matrix(generated_imgs, (self.n_checkpoint_rotations + 1), self.n_checkpoint_samples)

        img_output_dir = os.path.join(output_dir, "output_imgs")
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)

        cv2.imwrite(os.path.join(img_output_dir, str(step_number).zfill(6) + "_synth.jpg"), combined_image)
        with self.log_writer.as_default():
            tf.summary.image("generated_synth_images", combined_image[np.newaxis, :, :, [2, 1, 0]], step=step_number)

    # Start of checkpoint-related code
    def image_checkpoint(self, output_dir):
        step_number = self.get_training_step_number()

        latent = self._checkpoint_visualization_input["latent"]
        rotation = self._checkpoint_visualization_input["rotation"]

        generated_imgs = self.generate_images(latent, rotation)
        combined_image = confignet_utils.build_image_matrix(generated_imgs, self.n_checkpoint_rotations, self.n_checkpoint_samples)

        img_output_dir = os.path.join(output_dir, "output_imgs")
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)

        image_grid_file_path = os.path.join(img_output_dir, str(step_number).zfill(6) + ".png")
        cv2.imwrite(image_grid_file_path, combined_image)

        with self.log_writer.as_default():
            tf.summary.image("generated_images", combined_image[np.newaxis, :, :, [2, 1, 0]], step=step_number)

        self.synth_data_image_checkpoint(output_dir)

    def generate_output_for_metrics(self):
        return self.generate_images(self._generator_input_for_metrics["latent"], self._generator_input_for_metrics["rotation"])

    def run_checkpoints(self, output_dir, iteration_time, aml_run=None, checkpoint_start=None):
        checkpoint_start = time.clock()
        step_number = self.get_training_step_number()

        if step_number % self.config["image_checkpoint_period"] == 0:
            confignet_utils.log_loss_vals(self.synth_d_losses, output_dir, step_number, "synth_discriminator_",
                                        aml_run=aml_run, tb_log_writer=self.log_writer)
            confignet_utils.log_loss_vals(self.latent_d_losses, output_dir, step_number, "latent_discriminator_",
                                        aml_run=aml_run, tb_log_writer=self.log_writer)

        if checkpoint_start is None:
            checkpoint_start = time.clock()

        step_number = self.get_training_step_number()

        if step_number % self.config["metrics_checkpoint_period"] == 0:
            print("Running metrics")
            self.calculate_metrics(output_dir, aml_run=aml_run)
            checkpoint_output_dir = os.path.join(output_dir, "checkpoints")
            if not os.path.exists(checkpoint_output_dir):
                os.makedirs(checkpoint_output_dir)
            self.save(checkpoint_output_dir, str(step_number).zfill(6))

        if step_number % self.config["image_checkpoint_period"] == 0:
            self.image_checkpoint(output_dir)
            confignet_utils.log_loss_vals(self.g_losses, output_dir, step_number, "generator_", aml_run=aml_run, tb_log_writer=self.log_writer)
            confignet_utils.log_loss_vals(self.d_losses, output_dir, step_number, "discriminator_", aml_run=aml_run, tb_log_writer=self.log_writer)

            # Only actually display the checkpoint time if the checkpoint is run
            checkpoint_end = time.clock()

            checkpoint_time = checkpoint_end - checkpoint_start
            print("Training iteration time: %f"%(iteration_time))
            print("Checkpoint time: %f"%(checkpoint_time))

            if aml_run is not None:
                aml_run.log("Training iter time", iteration_time)
                aml_run.log("Checkpoint time", checkpoint_time)

            with self.log_writer.as_default():
                tf.summary.scalar("perf/training_iter_time", iteration_time, step=step_number)
                tf.summary.scalar("perf/checkpoint_time", checkpoint_time, step=step_number)


    def calculate_metrics(self, output_dir, aml_run=None):
        generated_images = self.generate_output_for_metrics()
        number_of_completed_iters = self.get_training_step_number()

        if "training_step_number" not in self.metrics.keys():
            self.metrics["training_step_number"] = []
        self.metrics["training_step_number"].append(number_of_completed_iters)
        self._inception_metric_object.update_and_log_metrics(generated_images, self.metrics, output_dir, aml_run, self.log_writer)



    # End of checkpoint-related code

    # Start of training code

    def update_smoothed_weights(self, smoother_alpha=0.999):
        training_weights = self.generator.get_weights()
        smoothed_weights = self.generator_smoothed.get_weights()

        for i in range(len(smoothed_weights)):
            smoothed_weights[i] = smoother_alpha * smoothed_weights[i] + (1 - smoother_alpha) * training_weights[i]

        self.generator_smoothed.set_weights(smoothed_weights)



    def sample_rotations(self, n_samples, axes=[0, 1, 2]):
        random_rotation = np.zeros((n_samples, 3))
        for axis in axes:
            random_rotation[:, axis] = np.pi * np.random.uniform(self.config["rotation_ranges"][axis][0], self.config["rotation_ranges"][axis][1], n_samples) / 180

        return random_rotation.astype(np.float32)

    def sample_latent_vector(self, n_samples):
        if self.config["latent_distribution"] == "normal":
            return np.random.normal(0, 1, (n_samples, self.config["latent_dim"]))
        elif self.config["latent_distribution"] == "uniform":
            return np.random.uniform(-1, 1, (n_samples, self.config["latent_dim"]))


    def sample_facemodel_params(self, n_samples):
        facemodel_params_from_distr = []
        for input_name in self.config["facemodel_inputs"].keys():
            facemodel_params_from_distr.append(self.facemodel_param_distributions[input_name].sample(n_samples)[0])

        return facemodel_params_from_distr

    def sample_synthetic_dataset(self, dataset, n_samples):
        sample_idxs = np.random.randint(0, dataset.imgs.shape[0], n_samples)
        facemodel_params = []
        for input_name in self.config["facemodel_inputs"].keys():
            facemodel_params.append(dataset.metadata_inputs[input_name][sample_idxs])
        render_rotations = dataset.metadata_inputs["rotations"][sample_idxs].astype(np.float32)

        gt_imgs = np.copy(dataset.imgs[sample_idxs]).astype(np.float32)
        eye_masks = np.copy(dataset.eye_masks[sample_idxs])

        return facemodel_params, render_rotations, gt_imgs, eye_masks


    def get_discriminator_batch(self, training_set):
        # Inputs
        img_idxs = np.random.randint(0, training_set.imgs.shape[0], self.get_batch_size())
        real_imgs = np.copy(training_set.imgs[img_idxs])
        real_imgs = real_imgs.astype(np.float32) / 127.5 - 1.0
        real_imgs = confignet_utils.flip_random_subset_of_images(real_imgs)
        real_imgs = tf.convert_to_tensor(real_imgs)

        latent_vector = self.sample_latent_vector(self.get_batch_size())
        random_rotation = self.sample_rotations(self.get_batch_size())
        fake_imgs = self.generator([latent_vector, random_rotation])

        return real_imgs, fake_imgs

    def get_synth_discriminator_batch(self, training_set):
        # Inputs
        img_idxs = np.random.randint(0, training_set.imgs.shape[0], self.get_batch_size())
        real_imgs = np.copy(training_set.imgs[img_idxs])
        real_imgs = real_imgs.astype(np.float32) / 127.5 - 1.0
        real_imgs = confignet_utils.flip_random_subset_of_images(real_imgs)
        real_imgs = tf.convert_to_tensor(real_imgs)

        facemodel_params, rotations, _, _ = self.sample_synthetic_dataset(training_set, self.get_batch_size())
        latent_vector = self.synthetic_encoder(facemodel_params)
        fake_imgs = self.generator([latent_vector, rotations])

        return real_imgs, fake_imgs

    def discriminator_training_step(self, training_set, optimizer):
        real_imgs, fake_imgs = self.get_discriminator_batch(training_set)

        with tf.GradientTape() as tape:
            losses = compute_discriminator_loss(self.discriminator, real_imgs, fake_imgs)

        trainable_weights = self.discriminator.trainable_weights
        gradients = tape.gradient(losses["loss_sum"], trainable_weights)
        optimizer.apply_gradients(zip(gradients, trainable_weights))

        return losses

    def synth_discriminator_training_step(self, synth_training_set, optimizer):
        real_imgs, fake_imgs = self.get_synth_discriminator_batch(synth_training_set)

        with tf.GradientTape() as tape:
            losses = compute_discriminator_loss(self.synth_discriminator, real_imgs, fake_imgs)

        trainable_weights = self.synth_discriminator.trainable_weights
        gradients = tape.gradient(losses["loss_sum"], trainable_weights)
        optimizer.apply_gradients(zip(gradients, trainable_weights))

        return losses

    def latent_discriminator_training_step(self, synth_training_set, optimizer):
        # Inputs
        real_latents = self.sample_latent_vector(self.get_batch_size())
        real_latents = tf.convert_to_tensor(real_latents, tf.float32)
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
        facemodel_params, synth_rotations, gt_imgs, eye_masks = self.sample_synthetic_dataset(synth_training_set, n_synth_in_batch)
        gt_imgs = gt_imgs.astype(np.float32) / 127.5 - 1.0

        # Real batch
        real_latents = self.sample_latent_vector(n_real_in_batch)
        real_rotations = self.sample_rotations(n_real_in_batch)

        losses = {}
        with tf.GradientTape() as tape:
            synth_latents = self.synthetic_encoder(facemodel_params)

            generator_output_synth = self.generator((synth_latents, synth_rotations))
            generator_output_real = self.generator((real_latents, real_rotations))

            losses["image_loss"] = self.config["image_loss_weight"] * self.perceptual_loss.loss(gt_imgs, generator_output_synth)
            losses["eye_loss"] = self.config["eye_loss_weight"] * eye_loss(gt_imgs, generator_output_synth, eye_masks)
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
            latent_discriminator_output = self.latent_discriminator(synth_latents)
            latent_gan_loss = GAN_G_loss(latent_discriminator_output)
            losses["latent_GAN_loss"] = self.config["domain_adverserial_loss_weight"] * latent_gan_loss

            # Latent regression loss start
            stacked_latent_vectors = tf.concat((synth_latents, real_latents), axis=0)
            stacked_generated_imgs = tf.concat((generator_output_synth, generator_output_real), axis=0)
            stacked_rotations = tf.concat((synth_rotations, real_rotations), axis=0)
            latent_regression_labels = tf.concat((stacked_latent_vectors, self.config["latent_regressor_rot_weight"] * stacked_rotations), axis=-1)

            # Regression of Z and rotation from output image
            latent_regression_loss = compute_latent_regression_loss(stacked_generated_imgs, latent_regression_labels, self.latent_regressor)
            losses["latent_regression_loss"] = self.config["latent_regression_weight"] * latent_regression_loss

            losses["loss_sum"] = tf.reduce_sum(list(losses.values()))

        trainable_weights = self.generator.trainable_weights + self.latent_regressor.trainable_weights + self.synthetic_encoder.trainable_weights
        gradients = tape.gradient(losses["loss_sum"], trainable_weights)
        optimizer.apply_gradients(zip(gradients, trainable_weights))

        return losses

    def setup_training(self, log_dir, synth_training_set, n_samples_for_metrics, real_training_set=None):
        if real_training_set is None:
            real_training_set = synth_training_set

        os.makedirs(log_dir, exist_ok=True)
        self.log_writer = tf.summary.create_file_writer(log_dir)

        self._inception_metric_object = InceptionMetrics(self.config, real_training_set)

        self._generator_input_for_metrics = {}
        self._generator_input_for_metrics["latent"] = self.sample_latent_vector(n_samples_for_metrics)
        self._generator_input_for_metrics["rotation"] = self.sample_rotations(n_samples_for_metrics)

        checkpoint_latent = self.sample_latent_vector(self.n_checkpoint_samples)
        checkpoint_latent = np.vstack([checkpoint_latent] * self.n_checkpoint_rotations)

        checkpoint_rotation = np.zeros((self.n_checkpoint_rotations, 3))
        checkpoint_rotation[:, 0] = np.pi * np.linspace(self.config["rotation_ranges"][0][0], self.config["rotation_ranges"][0][1], self.n_checkpoint_rotations) / 180
        checkpoint_rotation = np.hstack([checkpoint_rotation] * self.n_checkpoint_samples)
        checkpoint_rotation = np.reshape(checkpoint_rotation, (-1, 3))

        self._checkpoint_visualization_input = {}
        self._checkpoint_visualization_input["latent"] = checkpoint_latent
        self._checkpoint_visualization_input["rotation"] = checkpoint_rotation

        self.facemodel_param_distributions = synth_training_set.metadata_input_distributions

        facemodel_params, _, gt_imgs, _ = self.sample_synthetic_dataset(synth_training_set, self.n_checkpoint_samples)

        for i, param in enumerate(facemodel_params):
            facemodel_params[i] = np.tile(param, (self.n_checkpoint_rotations, 1))

        self._checkpoint_visualization_input["facemodel_params"] = facemodel_params
        self._checkpoint_visualization_input["gt_imgs"] = gt_imgs

    def train(self, real_training_set, synth_training_set, output_dir, log_dir, n_steps=100000, n_samples_for_metrics=1000, aml_run=None):
        self.setup_training(log_dir, synth_training_set, n_samples_for_metrics, real_training_set=real_training_set)
        start_step = self.get_training_step_number()

        discriminator_optimizer = keras.optimizers.Adam(**self.config["optimizer"])
        generator_optimizer = keras.optimizers.Adam(**self.config["optimizer"])

        for _ in range(start_step, n_steps):
            training_iteration_start = time.clock()

            for _ in range(self.config["n_discriminator_updates"]):
                d_loss = self.discriminator_training_step(real_training_set, discriminator_optimizer)
                synth_d_loss = self.synth_discriminator_training_step(synth_training_set, discriminator_optimizer)
                latent_d_loss = self.latent_discriminator_training_step(synth_training_set, discriminator_optimizer)

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

    ### End of training code


    ### Start of evaluation code

    def generate_images(self, latent_vector, rotations):
        generator_input_dict = self.generator.build_input_dict(latent_vector, rotations)
        imgs = self.generator_smoothed.predict(generator_input_dict)
        imgs = np.clip(imgs, -1.0, 1.0)
        imgs = ((imgs + 1) * 127.5).astype(np.uint8)

        return imgs

    def generate_images_from_facemodel(self, facemodel_params, rotations):
        latent_vectors = self.synthetic_encoder(facemodel_params)

        return self.generate_images(latent_vectors, rotations)

    def fit_facemodel_expression_params_to_latent(self, latent, unused_expr_idxs=None, param_name="blendshape_values",
                                                n_iters=2000, learning_rate=0.05, verbose=False):
        expression_idxs_in_latent = self.get_facemodel_param_idxs_in_latent(param_name)

        facemodel_param_names = list(self.config["facemodel_inputs"].keys())
        facemodel_param_dims = list(self.config["facemodel_inputs"].values())
        expression_idx_in_facemodel_params = list(facemodel_param_names).index(param_name)

        latent_exp_values = latent[:, expression_idxs_in_latent]
        facemodel_param_values = np.zeros((1, facemodel_param_dims[expression_idx_in_facemodel_params][0]), dtype=np.float32)
        facemodel_param_values = tf.Variable(facemodel_param_values)

        synthetic_encoder_model = self.synthetic_encoder.per_facemodel_input_mlps[param_name]

        optimizer = keras.optimizers.SGD(lr=learning_rate)

        for step in range(n_iters):
            with tf.GradientTape() as tape:
                predicted_latent = synthetic_encoder_model(facemodel_param_values)

                loss = tf.reduce_mean(tf.square(latent_exp_values - predicted_latent))

            gradients = tape.gradient(loss, [facemodel_param_values])
            optimizer.apply_gradients(zip(gradients, [facemodel_param_values]))
            facemodel_param_values.assign(tf.clip_by_value(facemodel_param_values, 0, 1))

            if unused_expr_idxs is not None:
                facemodel_param_values_numpy = facemodel_param_values.numpy()
                facemodel_param_values_numpy[:, unused_expr_idxs] = 0
                facemodel_param_values.assign(facemodel_param_values_numpy)

            if verbose:
                print("%d: %f"%(step, loss.numpy()))

        return facemodel_param_values.numpy()
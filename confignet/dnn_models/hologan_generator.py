# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tensorflow as tf
import tensorflow.keras as keras

from .building_blocks import Conv3dAdaIn, Conv2dAdaIn, AdaIn
from ..confignet_utils import euler_angles_to_matrix, transform_3d_grid_tf

import numpy as np


class HologanGenerator(tf.keras.models.Model):
    def __init__(self, latent_dim, output_shape, n_adain_mlp_units, n_adain_mlp_layers, gen_output_activation):
        super(HologanGenerator, self).__init__()

        self.output_img_shape = output_shape
        self.const_shape = (4, 4, 4, 512)
        n_features_in_first_layer = 256

        nl_f = keras.layers.LeakyReLU
        mlp_nl = lambda: keras.layers.LeakyReLU(alpha=0.2)

        self.zero_input = keras.layers.Lambda(lambda x: self.get_zero_inputs(x), name="zero_input")
        self.learned_input_layer = keras.layers.Dense(np.prod(self.const_shape),
                                                      kernel_initializer="zeros",
                                                      bias_initializer=tf.compat.v1.initializers.ones(),
                                                      name="learned_input")

        #pre-rotation function
        self.map_3d_0 = Conv3dAdaIn(
            num_feature_maps = n_features_in_first_layer, kernel_size = 3, double_conv = False,
            non_linear_after = None,
            z_size = latent_dim,
            mlp_num_units = n_adain_mlp_units,
            mlp_num_layers = n_adain_mlp_layers,
            mlp_non_linear = mlp_nl, conv_non_linear = nl_f)

        self.map_3d_1 = Conv3dAdaIn(
            num_feature_maps = n_features_in_first_layer // 2, kernel_size = 3, double_conv = False,
            non_linear_after = None,
            z_size = latent_dim,
            mlp_num_units = n_adain_mlp_units,
            mlp_num_layers = n_adain_mlp_layers,
            mlp_non_linear = mlp_nl, conv_non_linear = nl_f)

        #NOTE: rotation (done in call)

        #post-rotation function
        self.map_3d_post = keras.models.Sequential([
            keras.layers.Conv3D(n_features_in_first_layer // 4, 3, padding="same"),
            nl_f(),
            keras.layers.Conv3D(n_features_in_first_layer // 4, 3, padding="same"),
            nl_f()
        ])

        self.projection_conv = keras.layers.Conv2D(512, 1, activation=tf.nn.leaky_relu, padding="same")

        #2d mapping
        self.map_2d_0 = Conv2dAdaIn(
            num_feature_maps = n_features_in_first_layer, kernel_size = 4, double_conv = False,
            non_linear_after = None,
            z_size = latent_dim,
            mlp_num_units = n_adain_mlp_units,
            mlp_num_layers = n_adain_mlp_layers,
            mlp_non_linear = mlp_nl, conv_non_linear = nl_f)

        self.map_2d_1 = Conv2dAdaIn(
            num_feature_maps = n_features_in_first_layer // 4, kernel_size = 4, double_conv = False,
            non_linear_after = None,
            z_size = latent_dim,
            mlp_num_units = n_adain_mlp_units,
            mlp_num_layers = n_adain_mlp_layers,
            mlp_non_linear = mlp_nl, conv_non_linear = nl_f)

        self.map_2d_2 = Conv2dAdaIn(
            num_feature_maps = n_features_in_first_layer // 8, kernel_size = 4, double_conv = False,
            non_linear_after = None,
            z_size = latent_dim,
            mlp_num_units = n_adain_mlp_units,
            mlp_num_layers = n_adain_mlp_layers,
            mlp_non_linear = mlp_nl, conv_non_linear = nl_f)

        if self.output_img_shape[0] > 128:
            self.map_2d_2b = Conv2dAdaIn(
                num_feature_maps = n_features_in_first_layer // 8, kernel_size = 4, double_conv = False,
                non_linear_after = None,
                z_size = latent_dim,
                mlp_num_units = n_adain_mlp_units,
                mlp_num_layers = n_adain_mlp_layers,
                mlp_non_linear = mlp_nl, conv_non_linear = nl_f)

        if self.output_img_shape[0] > 256:
            self.map_2d_2c = Conv2dAdaIn(
                num_feature_maps = n_features_in_first_layer // 16, kernel_size = 4, double_conv = False,
                non_linear_after = None,
                z_size = latent_dim,
                mlp_num_units = n_adain_mlp_units,
                mlp_num_layers = n_adain_mlp_layers,
                mlp_non_linear = mlp_nl, conv_non_linear = nl_f)

        self.map_final = keras.layers.Conv2D(3, 4, activation=gen_output_activation, padding="same")

    def get_zero_inputs(self, input_layer):
        zero_input = tf.constant(0.0, shape = (1, 1), name='HoloGanConstantInput')
        zero_input = tf.tile(zero_input, (tf.shape(input_layer)[0], 1))

        return zero_input

    def build_input_dict(self, latent_vector, rotation):
        input_dict = {}
        if isinstance(latent_vector, list):
            input_dict["z_3d_0"] = latent_vector[0]
            input_dict["z_3d_1"] = latent_vector[1]

            input_dict["z_2d_0"] = latent_vector[2]
            input_dict["z_2d_1"] = latent_vector[3]
            input_dict["z_2d_2"] = latent_vector[4]
        else:
            input_dict["z_3d_0"] = latent_vector
            input_dict["z_3d_1"] = latent_vector

            input_dict["z_2d_0"] = latent_vector
            input_dict["z_2d_1"] = latent_vector
            input_dict["z_2d_2"] = latent_vector
        input_dict["rotation"] = rotation

        return input_dict

    def call(self, inputs):
        if not isinstance(inputs, dict):
            inputs = self.build_input_dict(inputs[0], inputs[1])

        zeros = self.zero_input(inputs['z_3d_0'])

        x = self.learned_input_layer(zeros)
        x = keras.layers.Reshape(self.const_shape, name="learned_input_reshape")(x)

        #upsample by factor of 2
        x = keras.layers.UpSampling3D()(x)

        #transform
        x = self.map_3d_0({'x': x, 'z' : inputs['z_3d_0']})
        x = keras.layers.UpSampling3D()(x)
        x = self.map_3d_1({'x': x, 'z' : inputs['z_3d_1']})

        #rotate in 3d
        transforms  = euler_angles_to_matrix(inputs['rotation'])
        x = transform_3d_grid_tf(x, transforms)

        #'rendering' layers
        x = self.map_3d_post(x)
        #...including the reshape
        x_s = list(x.shape)
        if x_s[0] is None:
            x_s[0] = -1
        x = tf.reshape(x, (x_s[0], x_s[1], x_s[2], x_s[3] * x_s[4]))
        x = self.projection_conv(x)

        x = self.map_2d_0({'x': x, 'z' : inputs['z_2d_0']})
        x = keras.layers.UpSampling2D()(x)
        x = self.map_2d_1({'x': x, 'z' : inputs['z_2d_1']})
        x = keras.layers.UpSampling2D()(x)
        x = self.map_2d_2({'x': x, 'z' : inputs['z_2d_2']})
        x = keras.layers.UpSampling2D()(x)
        if self.output_img_shape[0] > 128:
            x = self.map_2d_2b({'x': x, 'z' : inputs['z_2d_2']})
            x = keras.layers.UpSampling2D()(x)
        if self.output_img_shape[0] > 256:
            x = self.map_2d_2c({'x': x, 'z' : inputs['z_2d_2']})
            x = keras.layers.UpSampling2D()(x)

        x = self.map_final(x)

        return x

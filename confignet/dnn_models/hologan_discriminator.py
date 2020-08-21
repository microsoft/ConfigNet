# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

from .building_blocks import DiscrBlock

class HologanDiscriminator(tf.keras.models.Model):
    def __init__(self, img_shape, num_resample, disc_max_feature_maps, disc_kernel_size,
                       disc_expansion_factor, initial_from_rgb_layer_in_discr):
        super(HologanDiscriminator, self).__init__()

        self.initial_from_rgb_layer_in_discr = initial_from_rgb_layer_in_discr
        self.out_size = (int(img_shape[0] / np.power(2, num_resample)),
            int(img_shape[1] / np.power(2, num_resample)))

        if self.initial_from_rgb_layer_in_discr:
            self.initial_1x1_conv = keras.layers.Conv2D(3, 1, padding="same")

        e = 1
        max_feature_maps = lambda x : np.minimum(x, disc_max_feature_maps)

        self.conv_blocks = []
        self.style_classifiers = []

        for _ in range(num_resample):
            num_feature_maps = max_feature_maps(e * disc_expansion_factor)

            conv_block = DiscrBlock(num_feature_maps = num_feature_maps,
                kernel_size = disc_kernel_size, return_styles=True,
                conv_non_linear = keras.layers.LeakyReLU)
            style_classifier = keras.layers.Dense(1, input_shape=(2 * num_feature_maps,))

            self.conv_blocks.append(conv_block)
            self.style_classifiers.append(style_classifier)

            #expand number of feature maps
            e *= 2

        #calculate the number of input feature points
        self.num_linear_in = max_feature_maps(e * disc_max_feature_maps // 2) * self.out_size[0] * self.out_size[1]

        #dense layers standard gan loss (predicting true/false)
        self.disc_map = keras.layers.Dense(1, input_shape=(self.num_linear_in,), activation=None, use_bias=True)

    def call(self, input_img):
        if self.initial_from_rgb_layer_in_discr:
            x = self.initial_1x1_conv(input_img)
        else:
            x = input_img

        outputs = {}
        for i, (conv_block, style_classifier) in enumerate(zip(self.conv_blocks, self.style_classifiers)):
            x, style = conv_block(x)
            style_classifier_out = style_classifier(style)
            outputs["discr_style_" + str(i)] = style_classifier_out

        x = tf.reshape(x, (-1, tf.reduce_prod(x.shape[1:])))
        disc_prediction = self.disc_map(x)
        outputs["discr_final"] = disc_prediction

        return outputs

class HologanLatentRegressor(tf.keras.models.Model):
    def __init__(self, latent_dim, img_shape, num_resample, disc_max_feature_maps, disc_kernel_size,
                       disc_expansion_factor, initial_from_rgb_layer_in_discr):
        super(HologanLatentRegressor, self).__init__()

        self.initial_from_rgb_layer_in_discr = initial_from_rgb_layer_in_discr
        self.out_size = (int(img_shape[0] / np.power(2, num_resample)),
            int(img_shape[1] / np.power(2, num_resample)))

        #BUILD LAYERS-------------------------------------------------------------------------------------------
        if self.initial_from_rgb_layer_in_discr:
            self.initial_1x1_conv = keras.layers.Conv2D(3, 1, padding="same")

        e = 1 #tracking current expansion
        max_feature_maps = lambda x : np.minimum(x, disc_max_feature_maps)

        self.conv_blocks = []
        for _ in range(num_resample):
            num_feature_maps = max_feature_maps(e * disc_expansion_factor)

            conv_block = DiscrBlock(num_feature_maps = num_feature_maps,
                kernel_size = disc_kernel_size, return_styles=False,
                conv_non_linear = keras.layers.LeakyReLU)

            self.conv_blocks.append(conv_block)

            #expand number of feature maps
            e *= 2

        #calculate the number of input feature points
        self.num_linear_in = max_feature_maps(e * disc_max_feature_maps // 2) * self.out_size[0] * self.out_size[1]
        self.latent_predictor = keras.layers.Dense(latent_dim + 3, input_shape=(self.num_linear_in,), activation=None, use_bias=True)

    def call(self, inputs):
        img = inputs

        if self.initial_from_rgb_layer_in_discr:
            x = self.initial_1x1_conv(img)
        else:
            x = img

        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = tf.reshape(x, (-1, tf.reduce_prod(x.shape[1:])))

        latent_prediction = self.latent_predictor(x)

        return latent_prediction
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import resnet50

import numpy as np

class RealEncoder(tf.keras.models.Model):
    def __init__(self, latent_dim, input_shape, rotation_ranges):
        super(RealEncoder, self).__init__()

        self.resnet = resnet50.ResNet50(weights="imagenet", include_top=False, input_shape=input_shape, pooling="avg")

        self.resnet_feature_dim = np.prod(self.resnet.output.shape[1:])
        self.rotation_regressor = keras.layers.Dense(3, input_shape=(self.resnet_feature_dim,), activation=tf.nn.tanh)

        self.feature_to_latent_mlp = keras.layers.Dense(latent_dim, input_shape=(self.resnet_feature_dim,))

        self.rotation_range_multiplier = np.array([rotation_ranges[0][1], rotation_ranges[1][1], rotation_ranges[2][1]])
        self.rotation_range_multiplier = np.pi * self.rotation_range_multiplier / 180.0

    def call(self, input_img):
        input_img_0_255 = (input_img + 1) * 127.5
        preprocessed_img = resnet50.preprocess_input(input_img_0_255)

        resnet_features = self.resnet(preprocessed_img)

        raw_rotation = self.rotation_regressor(resnet_features)
        scaled_rotation = self.rotation_range_multiplier * raw_rotation

        embedding = self.feature_to_latent_mlp(resnet_features)

        return embedding, scaled_rotation

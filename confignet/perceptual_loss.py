# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Tuple

class PerceptualLoss():
    def __init__(self, input_shape: Tuple[int, int, int], model_type="imagenet"):
        self.input_shape = input_shape
        self.model_type = model_type

        if self.model_type == "VGGFace":
            self._pretrained_dnn_activations = self._get_pretrained_dnn_vggface()
        elif self.model_type == "imagenet":
            self._pretrained_dnn_activations = self._get_pretrained_dnn_imagenet()

    def _get_pretrained_dnn_imagenet(self):
        pretrained_dnn = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=self.input_shape, pooling=None)

        idxs_of_used_layers = [1, 2, 8, 13]
        used_layer_activations = [pretrained_dnn.layers[layer_idx].output for layer_idx in idxs_of_used_layers]

        layer_activations = keras.models.Model(inputs=pretrained_dnn.input, outputs=used_layer_activations)

        return layer_activations

    def _get_pretrained_dnn_vggface(self):
        # Pretrained model from the keras-vggface repo
        weights_path = keras.utils.get_file("rcmalli_vggface_tf_notop_vgg16.h5",
                        "https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_vgg16.h5",
                        cache_subdir="models")
        pretrained_dnn = keras.applications.vgg16.VGG16(include_top=False, input_shape=self.input_shape)
        pretrained_dnn.load_weights(weights_path, by_name=True)

        idxs_of_used_layers = [1, 2, 8, 12]
        used_layer_activations = [pretrained_dnn.layers[layer_idx].output for layer_idx in idxs_of_used_layers]

        layer_activations = keras.models.Model(inputs=pretrained_dnn.input, outputs=used_layer_activations)

        return layer_activations

    def loss(self, predicted, data):
        terms = self._loss_terms(predicted, data)

        sum_of_terms = 0
        for term in terms:
            sum_of_terms += term

        return tf.reduce_mean(sum_of_terms)

    def _preprocess_input(self, input_img):
        preprocessed_input = (input_img  + 1) * 127.5
        if self.model_type == "VGGFace":
            # Mean values of face images from the VGGFace paper
            preprocessed_input = preprocessed_input - (93.5940, 104.7624, 129.1863)
        elif self.model_type == "imagenet":
            preprocessed_input = keras.applications.vgg19.preprocess_input(preprocessed_input)


        return preprocessed_input

    def _loss_terms(self, predicted, data):
        if len(predicted.shape) == 3:
            predicted = tf.expand_dims(predicted, 0)
        if len(data.shape) == 3:
            data = tf.expand_dims(data, 0)

        preprocessed_predicted = self._preprocess_input(predicted)
        preprocessed_data = self._preprocess_input(data)

        all_activations_predicted = self._pretrained_dnn_activations(preprocessed_predicted)
        all_activations_data = self._pretrained_dnn_activations(preprocessed_data)

        loss_terms = []
        for activations_predicted, activations_data in zip(all_activations_predicted, all_activations_data):
            activations_predicted = tf.reshape(activations_predicted, [-1])
            activations_data = tf.reshape(activations_data, [-1])

            loss_terms.append(tf.reduce_mean(tf.losses.mean_squared_error(activations_predicted, activations_data)))

        return loss_terms

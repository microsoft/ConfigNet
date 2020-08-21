# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tensorflow as tf
import tensorflow.keras as keras
from .. import confignet_utils

# This will be replaced by keras.layers.layernormalization
from .instance_normalization import InstanceNormalization

#Block of 3d Convolution(s), followed by instance norm
class Conv3dAdaIn(tf.keras.models.Model):
    def __init__(self,
            num_feature_maps, kernel_size, double_conv,
            non_linear_after,
            z_size,
            mlp_num_units,
            mlp_num_layers,
            mlp_non_linear = keras.layers.LeakyReLU, conv_non_linear = keras.layers.LeakyReLU):
        super(Conv3dAdaIn, self).__init__()

        if double_conv:
            self.map_3d = keras.models.Sequential([
                keras.layers.Conv3D(num_feature_maps, kernel_size, padding="same"),
                conv_non_linear(),
                keras.layers.Conv3D(num_feature_maps, kernel_size, padding="same"),
            ])
        else:
            self.map_3d = keras.models.Sequential([
                keras.layers.Conv3D(num_feature_maps, kernel_size, padding="same"),
            ])

        self.adain = AdaIn(non_linear_after = None, z_size = z_size, mlp_num_units = mlp_num_units,
            num_features = num_feature_maps, mlp_num_layers = mlp_num_layers, mlp_non_linear = mlp_non_linear)

        self.nl = conv_non_linear()

    def call(self, inputs):
        x = inputs['x']
        z = inputs['z']

        x = self.map_3d(x)
        x = self.nl(x)
        x = self.adain(x, z)
        return x

#Block of 2d Convolution(s), followed by instance norm
class Conv2dAdaIn(tf.keras.models.Model):
    def __init__(self,
            num_feature_maps, kernel_size, double_conv,
            non_linear_after,
            z_size,
            mlp_num_units,
            mlp_num_layers,
            mlp_non_linear = keras.layers.LeakyReLU, conv_non_linear = keras.layers.LeakyReLU):
        super(Conv2dAdaIn, self).__init__()

        if double_conv:
            self.map_2d = keras.models.Sequential([
                keras.layers.Conv2D(num_feature_maps, kernel_size, padding="same"),
                conv_non_linear(),
                keras.layers.Conv2D(num_feature_maps, kernel_size, padding="same"),
            ])
        else:
            self.map_2d = keras.models.Sequential([
                keras.layers.Conv2D(num_feature_maps, kernel_size, padding="same"),
            ])

        self.adain = AdaIn(non_linear_after = None, z_size = z_size, mlp_num_units = mlp_num_units,
            num_features = num_feature_maps, mlp_num_layers = mlp_num_layers, mlp_non_linear = mlp_non_linear)

        self.nl = conv_non_linear()

    def call(self, inputs):
        x = inputs['x']
        z = inputs['z']

        x = self.map_2d(x)
        x = self.nl(x)
        x = self.adain(x, z)
        return x

#Block of 2d Convolution(s) with Instance Norm that optionally also returns the activation "style"
class DiscrBlock(tf.keras.models.Model):
    def __init__(self,
            num_feature_maps, kernel_size, return_styles,
            conv_non_linear = keras.layers.LeakyReLU):
        super(DiscrBlock, self).__init__()

        self.return_styles = return_styles

        self.map_2d = keras.layers.Conv2D(num_feature_maps, kernel_size, padding="same", strides=2)
        # TODO: change this to keras.layers.LayerNorm, this custom class is used ATM to be able to load weights from TF 1.13.
        self.instance_norm = InstanceNormalization(axis=-1)

        self.nl = conv_non_linear()

    def call(self, inputs):
        x = self.map_2d(inputs)

        if self.return_styles:
            styles = tf.concat(confignet_utils.get_layer_style(x), -1)
            styles = tf.reshape(styles, (-1, tf.reduce_prod(styles.shape[1:])))


        x = self.nl(x)
        x = self.instance_norm(x)

        if self.return_styles:
            return x, styles
        else:
            return x

#Adaptive Instance Norm
class AdaIn(tf.keras.models.Model):
    def __init__(self, non_linear_after,
            z_size,
            mlp_num_units,
            num_features,
            mlp_num_layers, mlp_non_linear = keras.layers.LeakyReLU):
        super(AdaIn, self).__init__()
        self.num_features = num_features
        #construct the layers feeding into the AdaIn
        self.adain_mlp = MLPSimple(num_layers = mlp_num_layers,
            num_in = z_size, num_hidden = mlp_num_units, num_out = num_features * 2,
            non_linear = mlp_non_linear, non_linear_last = None)

        if non_linear_after is not None:
            self.nl = keras.layers.LeakyReLU()
        else:
            self.nl = None

        self.instance_norm_3d = keras.layers.LayerNormalization(axis=[1, 2, 3], center=False, scale=False)
        self.instance_norm_2d = keras.layers.LayerNormalization(axis=[1, 2], center=False, scale=False)

    def call(self, x, z):
        #get adaptive instance norm parameters
        if len(x.shape) == 5: #3d input
            z = tf.reshape(self.adain_mlp(z), [-1, 2, 1, 1, 1, self.num_features])
            x = self.instance_norm_3d(x)
        else: #2d input
            z = tf.reshape(self.adain_mlp(z), [-1, 2, 1, 1, self.num_features])
            x = self.instance_norm_2d(x)
        #now apply instance norm
        x = x * (z[:, 0, :] + 1) + z[:, 1, :]

        if self.nl is not None:
            x = self.nl(x)

        return x

#Constructs a generic MLP
class MLPSimple(tf.keras.models.Model):
    def __init__(self, num_layers, num_in, num_hidden, num_out, non_linear, non_linear_last):
        super(MLPSimple, self).__init__()

        self.map = keras.models.Sequential()
        self.num_in = num_in
        self.num_out = num_out

        current_num_in = num_in
        #hidden layers
        for _ in range(num_layers - 1):
            self.map.add(tf.keras.layers.Dense(num_hidden, input_shape=(current_num_in,)))
            self.map.add(non_linear())
            current_num_in = num_hidden

        #last layer
        self.map.add(tf.keras.layers.Dense(num_out, input_shape=(current_num_in,)))
        if non_linear_last is not None:
            self.map.add(non_linear_last())

    def call(self, inputs):
        return self.map(inputs)

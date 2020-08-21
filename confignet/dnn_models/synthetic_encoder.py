# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tensorflow as tf
import tensorflow.keras as keras

from .building_blocks import MLPSimple

from collections import OrderedDict

class SyntheticDataEncoder(tf.keras.models.Model):
    def __init__(self, synthetic_encoder_inputs, num_layers):
        super(SyntheticDataEncoder, self).__init__()

        assert isinstance(synthetic_encoder_inputs, OrderedDict)
        self.facemodel_param_names = list(synthetic_encoder_inputs.keys())

        self.per_facemodel_input_mlps = {}

        for input_name in self.facemodel_param_names:
            input_dim = synthetic_encoder_inputs[input_name][0]
            output_dim = synthetic_encoder_inputs[input_name][1]

            mlp = MLPSimple(num_layers=num_layers,
                        num_in=input_dim,
                        num_hidden=input_dim,
                        num_out=output_dim,
                        non_linear=keras.layers.LeakyReLU,
                        non_linear_last=None)
            self.per_facemodel_input_mlps[input_name] = mlp

            # The MLP needs to be added to the object for its weights to be trainable
            mlp_name = "mlp_" + input_name
            self.__setattr__(mlp_name, mlp)

    def build_input_dictionary(self, inputs):
        # If input is a list then we just convert it to a dict
        if isinstance(inputs, list):
            return dict(zip(self.facemodel_param_names, inputs))

        # Otherwise we need to divide the vector to individual inputs
        input_dict = {}
        used_dim_sum = 0
        for input_name, mlp in self.per_facemodel_input_mlps.items():
            input_dim = mlp.num_in
            input_dict[input_name] = tf.gather(inputs, tf.range(used_dim_sum, used_dim_sum + input_dim), axis=1)
            used_dim_sum += input_dim

        return input_dict

    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            inputs = self.build_input_dictionary(inputs)

        output = OrderedDict()
        for facemodel_param_name in self.facemodel_param_names:
            output[facemodel_param_name] = self.per_facemodel_input_mlps[facemodel_param_name](inputs[facemodel_param_name])

        concatenated_output = tf.concat(list(output.values()), axis=1)

        return concatenated_output
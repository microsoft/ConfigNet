# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
""" Scripts containing common code for training """
import numpy as np
import tensorflow as tf
import random

def initialize_random_seed(seed):
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed )
    random.seed(seed)

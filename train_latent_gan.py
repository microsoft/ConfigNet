# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import sys
import argparse

import confignet
from confignet.latent_gan import DEFAULT_CONFIG
import training_utils

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--confignet_path", default=None, required=True,
                        help="Path to a confignet model that will be used to train the latent gan")
    parser.add_argument("--training_set_path", help="Path to the training set", default=None, required=True)
    parser.add_argument("--output_dir", help="Directory where training output will be stored", default=None, required=True)
    parser.add_argument("--num_mlp_layers", type=int, help="Number of discriminator and generator layers", default=DEFAULT_CONFIG["num_mlp_layers"])
    parser.add_argument("--hidden_layer_size_multiplier", type=float, help="Multiplier for number of units in hidden layers",
                        default=DEFAULT_CONFIG["hidden_layer_size_multiplier"])
    parser.add_argument("--latent_distribution_type", help="Type of distribution, normal or uniform", default=DEFAULT_CONFIG["latent_distribution_type"])
    parser.add_argument("--batch_size", type=int, help="Batch size used in training", default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--n_training_steps", type=int, help="Number of training steps that will be run", default=100000),
    parser.add_argument("--n_samples_for_metrics", type=int, help="Number of samples used in training-time metrics", default=1000)
    parser.add_argument("--data_dir", help="Optional path to which the dataset paths are appended", default=None)
    parser.add_argument("--log_dir", help="Path to which the tensorboard logs will be written, defaults to output_dir", default=None)
    args = parser.parse_args(args)

    training_utils.initialize_random_seed(0)

    if args.data_dir is not None:
        args.training_set_path = os.path.join(args.data_dir, args.training_set_path)
        args.confignet_path = os.path.join(args.data_dir, args.confignet_path)
    if args.log_dir is None:
        args.log_dir = args.output_dir

    training_set = confignet.NeuralRendererDataset.load(args.training_set_path)
    confignet_model = confignet.load_confignet(args.confignet_path)

    config = {}
    config["latent_dim"] = confignet_model.config["latent_dim"]
    config["num_mlp_layers"] = args.num_mlp_layers
    config["latent_distribution_type"] = args.latent_distribution_type
    config["hidden_layer_size_multiplier"] = args.hidden_layer_size_multiplier
    config["batch_size"] = args.batch_size
    config["n_samples_for_metrics"] = args.n_samples_for_metrics

    latent_gan = confignet.LatentGAN(config)
    latent_gan.train(training_set, confignet_model, args.output_dir, args.log_dir, n_iters=args.n_training_steps)

if __name__ == "__main__":
    parse_args(sys.argv[1:])
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script for starting the training of the ConfigNet"""

import numpy as np
import argparse
import os
import sys
import json

import training_utils
import confignet
from confignet.confignet_first_stage import DEFAULT_CONFIG

def parse_args(args):
    parser = argparse.ArgumentParser(description="ConfigNet training")
    parser.add_argument("--output_dir", help="Path to the directory where the output will be stored", required=True)
    parser.add_argument("--log_dir", help="Directory where tensorboard logs will be written", default=None)
    parser.add_argument("--data_dir", help="Optional path to which the dataset paths are appended", default=None)

    parser.add_argument("--real_training_set_path", help="Path to the real training set file", required=True)
    parser.add_argument("--synth_training_set_path", help="Path to the synthetic training set file", required=True)
    parser.add_argument("--validation_set_path", help="Path to the validation set file", required=True)
    parser.add_argument("--attribute_classifier_path", help="Path to attribute classifier that will be used in metrics", required=True)
    parser.add_argument("--batch_size", type=int, help="Batch size used in training ", default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--stage_1_training_steps", type=int, help="Number of training steps in first stage training", default=50000)
    parser.add_argument("--stage_2_training_steps", type=int, help="Number of training steps in second stage training", default=100000)
    parser.add_argument("--n_samples_for_metrics", type=int, help="Number of samples used in training-time metrics", default=1000)
    args = parser.parse_args(args)

    aml_run = confignet.azure_ml_utils.get_aml_run()
    confignet.azure_ml_utils.log_job_params(aml_run, args)

    training_utils.initialize_random_seed(0)

    if args.data_dir is not None:
        args.real_training_set_path = os.path.join(args.data_dir, args.real_training_set_path)
        args.synth_training_set_path = os.path.join(args.data_dir, args.synth_training_set_path)
        args.validation_set_path = os.path.join(args.data_dir, args.validation_set_path)
        args.attribute_classifier_path = os.path.join(args.data_dir, args.attribute_classifier_path)
    if args.log_dir is None:
        args.log_dir = args.output_dir
    real_training_set = confignet.NeuralRendererDataset.load(args.real_training_set_path)
    synth_training_set = confignet.NeuralRendererDataset.load(args.synth_training_set_path)
    validation_set = confignet.NeuralRendererDataset.load(args.validation_set_path)

    # if checkpoint not loaded
    config = {
        "batch_size": args.batch_size,
        "output_shape": real_training_set.imgs.shape[1:]
    }
    config = confignet.confignet_utils.merge_configs(DEFAULT_CONFIG, config)
    synth_training_set.process_metadata(config, True)


    ### first stage training
    first_stage_model = confignet.ConfigNetFirstStage(config)
    first_stage_output_dir = os.path.join(args.output_dir, "first_stage")
    first_stage_model.train(real_training_set, synth_training_set, first_stage_output_dir,
                            args.log_dir, n_steps=args.stage_1_training_steps,
                            n_samples_for_metrics=args.n_samples_for_metrics, aml_run=aml_run)

    first_stage_weights = first_stage_model.get_weights()
    ### end of first stage training

    ### second stage training
    config["image_loss_weight"] *= 10 # increase image loss weight
    second_stage_model = confignet.ConfigNet(config)
    confignet.ConfigNetFirstStage.set_weights(second_stage_model, first_stage_weights)

    second_stage_model.train(real_training_set, synth_training_set, validation_set, args.attribute_classifier_path,
                             args.output_dir, args.log_dir, n_steps=args.stage_1_training_steps,
                             n_samples_for_metrics=args.n_samples_for_metrics, aml_run=aml_run)

if __name__ == "__main__":
    parse_args(sys.argv[1:])
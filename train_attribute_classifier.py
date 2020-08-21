# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import sys
import argparse
import numpy as np

from confignet import NeuralRendererDataset
from confignet.metrics.celeba_attribute_prediction import CelebaAttributeClassifier, DEFAULT_CONFIG

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_set_path", help="Path to the training set", required=True)
    parser.add_argument("--validation_set_path", help="Path to the validation set", required=True)
    parser.add_argument("--output_dir", help="Path to the output directory", required=True)
    parser.add_argument("--n_epochs", type=int, help="Number of epochs", default=1000)
    parser.add_argument("--steps_per_epoch", type=int, help="Number of steps per epoch", default=100)
    parser.add_argument("--batch_size", type=int, help="Batch size that will be used for training", default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--ignored_attributes", nargs="+", help="Attributes that will not be used", default=["Wearing_Necklace", "Wearing_Necktie"])
    args = parser.parse_args(args)

    training_set = NeuralRendererDataset.load(args.training_set_path)
    validation_set = NeuralRendererDataset.load(args.validation_set_path)

    config = DEFAULT_CONFIG
    config["input_shape"] = training_set.imgs.shape[1:]
    config["batch_size"] = args.batch_size

    predicted_attributes = [attribute for attribute in training_set.attributes[0].keys() if attribute not in args.ignored_attributes]
    config["predicted_attributes"] = sorted(predicted_attributes)

    np.random.seed(0)
    celeba_classifier = CelebaAttributeClassifier(config)
    celeba_classifier.train(training_set, validation_set, args.output_dir, n_epochs=args.n_epochs, steps_per_epoch=args.steps_per_epoch)

if __name__ == "__main__":
    parse_args(sys.argv[1:])
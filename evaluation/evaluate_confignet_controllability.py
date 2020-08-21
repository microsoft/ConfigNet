# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import sys
import argparse
import json
from matplotlib import pyplot as plt
import numpy as np

import evaluation_utils

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import confignet


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to the confignet model",
                        default=os.path.join(os.path.dirname(__file__), "..", "models", "confignet_256", "model.json"))
    parser.add_argument("--test_set_path", help="Path to the test set", required=True)
    parser.add_argument("--output_dir", help="Directory where results will be written", required=True)
    parser.add_argument("--attribute_classifier_path", help="Path to the celeba attribute classifier that will be used for testing",
                        default=os.path.join(os.path.dirname(__file__), "..", "models", "attribute_classifier", "model.json"))
    parser.add_argument("--n_fine_tuning_iters", type=int, help="Number of fine tuning iterations that will be performed on each image", default=0)
    parser.add_argument("--n_samples", type=int, help="Number of samples used for testing", default=1000)
    parser.add_argument("--write_images", help="Number of samples used for testing", action="store_true", default=False)
    args = parser.parse_args(args)

    if args.model_path is None:
        args.model_path = evaluation_utils.dnn_filename_prompt()

    confignet_model = confignet.load_confignet(args.model_path)
    test_set = confignet.NeuralRendererDataset.load(args.test_set_path)
    test_imgs = test_set.imgs[:args.n_samples]

    metrics_extractor = confignet.ControllabilityMetrics(confignet_model, args.attribute_classifier_path,
                                                       per_image_tuning_iters=args.n_fine_tuning_iters)

    metrics_filename = "contr_metrics_tuning_iters_%d_"%(args.n_fine_tuning_iters)
    metrics_filename += os.path.splitext(os.path.basename(args.model_path))[0]
    if args.write_images:
        img_output_dir = os.path.join(args.output_dir, metrics_filename)
    else:
        img_output_dir = None
    metrics = metrics_extractor.get_metrics(test_imgs, img_output_dir=img_output_dir)

    set_attribute_values = [x[0] for x in list(metrics.values()) if isinstance(x, tuple)]
    not_set_attribute_values = [x[1] for x in list(metrics.values()) if isinstance(x, tuple)]
    other_attr_deltas = [x[2] for x in list(metrics.values()) if isinstance(x, tuple)]
    correlation_coefficient = [x[3] for x in list(metrics.values()) if isinstance(x, tuple)]
    tick_labels = [key for key in metrics.keys() if isinstance(metrics[key], tuple)]

    plt.figure(figsize=(12, 9))
    plt.plot(set_attribute_values)
    plt.plot(not_set_attribute_values)
    plt.plot(other_attr_deltas)
    plt.plot(correlation_coefficient)
    plt.legend(["Attribute value for I_+", "Attribute value for I_-", "Mean difference of other attributes", "Corr coef"])

    plt.xticks(range(len(set_attribute_values)), rotation=45)
    plt.gca().set_xticklabels(tick_labels)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, metrics_filename + ".png"))

    with open(os.path.join(args.output_dir, metrics_filename + ".json"), "w") as fp:
        json.dump(metrics, fp, indent=4)

    csv_content = np.vstack((set_attribute_values, not_set_attribute_values, other_attr_deltas, correlation_coefficient))
    np.savetxt(os.path.join(args.output_dir, metrics_filename + ".csv"), csv_content, delimiter=",")

if __name__ == "__main__":
    parse_args(sys.argv[1:])

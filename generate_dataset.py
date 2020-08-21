# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script for generating a dataset for neural renderer training"""
import os
import sys
import argparse
import numpy as np

import confignet

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Script for generating avatar datasets")
    parser.add_argument("--dataset_dir", help="Path to the directory containing the dataset images", required=True)
    parser.add_argument("--dataset_name", help="Name for the output dataset file", required=True)
    parser.add_argument("--output_dir", help="Path to the output directory, the .npz file name will correspond to the dataset directory name", required=True)
    parser.add_argument("--img_size", type=int, help="Size of the image, same number will be used for height and width", default=256)
    parser.add_argument("--pre_normalize", type=int, help="If set to 0 pre_normalization will not be performed", default=1)
    parser.add_argument("--img_output_dir", help="If this directory is specified the aligned face images will be dumped to it", default=None)
    parser.add_argument("--load_attributes", help="If specified, the script will look for a celeba attribute file in the dataset_dir", action="store_true", default=False)
    parser.add_argument("--synthetic_data", help="If specified the dataset will require an accompanying .json metadata file for each image", action="store_true", default=False)
    args = parser.parse_args(argv)

    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    img_size = args.img_size
    img_output_dir = args.img_output_dir
    synthetic_data = args.synthetic_data
    load_attributes = args.load_attributes

    dataset = confignet.NeuralRendererDataset((img_size, img_size, 3), synthetic_data)

    dataset_name = dataset_name + "_res_" + str(img_size)
    output_path = os.path.join(output_dir, dataset_name + ".pck")
    os.makedirs(output_dir, exist_ok=True)

    if load_attributes:
        attribute_file_path = os.path.join(dataset_dir, "list_attr_celeba.txt")
    else:
        attribute_file_path = None

    dataset.generate_face_dataset(dataset_dir, output_path, attribute_label_file_path=attribute_file_path, pre_normalize=args.pre_normalize == 1)
    if img_output_dir is not None:
        print("Writing aligned images to %s"%(img_output_dir))
        dataset.write_images(img_output_dir)
        if load_attributes:
            dataset.write_images_by_attribute(img_output_dir)

if __name__ == "__main__":
    parse_args(sys.argv[1:])
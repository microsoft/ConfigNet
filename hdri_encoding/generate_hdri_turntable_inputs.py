# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Script that rotates an HDR environment image, embeds each rotated version into the space of a PCA model and saves the output.
The output can then be used to visualize light source rotation using ConfigNet.
"""

import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

from hdri_pca_model import HDRIModelPCA


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdri_file_path", help="Path to the hdri image that will be used", default=None, required=True)
    parser.add_argument("--output_file_path", help="Path to the which the embeddings will be saved",
                        default=os.path.join(os.path.dirname(__file__), "..", "assets", "hdri_turntable_embeddings.npy"))
    parser.add_argument("--hdri_model_path", help="Path to the hdri PCA model",
                        default=os.path.join(os.path.dirname(__file__), "..", "assets", "hdri_model_20190919.pck"))
    parser.add_argument("--n_hdri_rotations", type=int, help="Number of rotated embeddings to generate", default=90)
    parser.add_argument("--hdri_output_dir", default=None,
                        help="Path to image output dir, if specified the encoded HDRIs will be decoded and saved to that directory")
    args = parser.parse_args(args)


    hdri = cv2.imread(args.hdri_file_path, -1)
    hdri_rotations = np.linspace(-180, 180, args.n_hdri_rotations)
    stacked_hdris = np.stack([hdri] * args.n_hdri_rotations)

    hdri_model = HDRIModelPCA.load(args.hdri_model_path)
    embeddings = hdri_model.transform(stacked_hdris, hdri_rotations)
    np.save(args.output_file_path, embeddings)

    if args.hdri_output_dir is not None:
        for i in range(args.n_hdri_rotations):
            reconstructed_hdri = hdri_model.inverse_transform(embeddings[[i]])[0]
            reconstructed_hdri = np.clip(reconstructed_hdri[:, :, [2, 1, 0]], 0, 1)
            plt.imshow(reconstructed_hdri)
            plt.savefig(os.path.join(args.hdri_output_dir, str(i).zfill(4) + ".jpg"))
            plt.clf()

if __name__ == "__main__":
    parse_args(sys.argv[1:])
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script for adding hdmi embeddings to render metadata .yaml files"""

import os
import argparse
import json
import glob
import numpy as np
from matplotlib import pyplot as plt

from hdri_pca_model import HDRIModelPCA, load_hdris, rotate_hdri
from hdri_encoding_utils import load_metadata_dicts, save_metadata_dicts

def update_metadata_dicts(metadata_dicts, hdri_embeddings):
    for i, _ in enumerate(metadata_dicts):
        metadata_dicts[i]["hdri_embedding"] = hdri_embeddings[i].tolist()

    return metadata_dicts

def draw_hdri_pair(hdri_1, hdri_2):
    hdri_1_processed = np.clip(hdri_1[:, :, [2, 1, 0]], 0, 1)
    hdri_2_processed = np.clip(hdri_2[:, :, [2, 1, 0]], 0, 1)

    plt.subplot(121)
    plt.imshow(hdri_1_processed)
    plt.subplot(122)
    plt.imshow(hdri_2_processed)

def get_hdri_embeddings(hdri_model, hdris, hdri_names, metadata_dicts, hdri_output_dir=None):
    hdri_embeddings = []

    for i, metadata_dict in enumerate(metadata_dicts):
        if i % 100 == 0:
            print(i)
        hdri_name = metadata_dict["illumination"]["HDRI_filename"]
        hdri_rotation = 180 * metadata_dict["illumination"]["HDRI_rotation"][2] / np.pi
        hdri_idx = hdri_names.index(hdri_name)
        hdri = hdris[hdri_idx]

        embedding = hdri_model.transform(hdri[np.newaxis], [hdri_rotation])[0]

        if hdri_output_dir is not None:
            reconstructed_hdri = hdri_model.inverse_transform(embedding[np.newaxis])[0]
            original_hdri_rotated = rotate_hdri(hdri, hdri_rotation)

            draw_hdri_pair(original_hdri_rotated, reconstructed_hdri)
            plt.savefig(os.path.join(hdri_output_dir, str(i).zfill(4) + ".jpg"))
            plt.clf()

        hdri_embeddings.append(embedding)

    return np.array(hdri_embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for adding hdmi embeddings to render metadata .yaml files")
    parser.add_argument("--input_dir", help="Directory containing the render metadata to be updated", required=True)
    parser.add_argument("--render_asset_dir", help="Path to render asset directory", required=True)
    parser.add_argument("--hdri_output_dir", help="Output directory to which the hdris and their reconstru will be written", default=None)
    parser.add_argument("--model_path", help="Path to the HDRI model that will be used to generate the embeddings",
                        default=os.path.join(os.path.dirname(__file__), "..", "assets", "hdri_model_20200116.pck"))

    args = parser.parse_args()

    hdri_model = HDRIModelPCA.load(args.model_path)

    metadata_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    metadata_dicts = load_metadata_dicts(metadata_files)

    hdri_dir = os.path.join(args.render_asset_dir, "HDRI")
    hdris, hdri_paths = load_hdris(hdri_dir)
    hdri_names = [os.path.basename(path) for path in hdri_paths]

    hdri_embeddings = get_hdri_embeddings(hdri_model, hdris, hdri_names, metadata_dicts, args.hdri_output_dir)

    metadata_dicts = update_metadata_dicts(metadata_dicts, hdri_embeddings)
    save_metadata_dicts(metadata_dicts, metadata_files)
""" Test of the HDRI image processing code """
import os
import sys
import shutil
import glob
import numpy as np
import pytest
from fixtures import temporary_output_dir, test_asset_dir

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "hdri_encoding"))
import hdri_pca_model
import generate_hdri_turntable_inputs

def test_hdri_model_creation_script(temporary_output_dir, test_asset_dir):
    hdri_dir = os.path.join(test_asset_dir, "hdri_encoding")
    base_script_arguments = "--hdri_dir %s --output_dir %s --write_hdris --output_shape %d %d"%(hdri_dir, temporary_output_dir, 10, 20)

    # Test specifying components by number
    script_arguments = base_script_arguments + " --n_components 5"
    hdri_pca_model.parse_args(script_arguments.split(" "))

    # Test specifying components by variance fraction
    script_arguments = base_script_arguments + " --n_components 0.9"
    hdri_pca_model.parse_args(script_arguments.split(" "))

def test_hdri_transform(test_asset_dir):
    hdri_model_path = os.path.join(test_asset_dir, "hdri_encoding", "hdri_model.pck")
    hdri_dir = os.path.join(test_asset_dir, "hdri_encoding")

    hdri_model = hdri_pca_model.HDRIModelPCA.load(hdri_model_path)
    hdri_images, _ = hdri_pca_model.load_hdris(hdri_dir)

    transformed = hdri_model.transform(hdri_images)
    re_generated = hdri_model.inverse_transform(transformed)

    re_transformed = hdri_model.transform(re_generated)
    re_re_generated = hdri_model.inverse_transform(re_transformed)

    assert np.all(np.isclose(re_transformed, transformed, atol=1e-6))
    assert np.all(np.isclose(re_re_generated, re_generated, atol=1e-6))

def test_hdri_turntable_generation(test_asset_dir, temporary_output_dir):
    hdri_path = os.path.join(test_asset_dir, "hdri_encoding", "001.hdr")
    output_path = os.path.join(temporary_output_dir, "test.npy")
    hdri_model_path = os.path.join(test_asset_dir, "hdri_encoding", "hdri_model.pck")

    script_arguments = "--hdri_file_path %s --output_file_path %s"%(hdri_path, output_path)
    script_arguments += " --hdri_model_path %s"%hdri_model_path

    generate_hdri_turntable_inputs.parse_args(script_arguments.split(" "))

    assert os.path.exists(output_path)

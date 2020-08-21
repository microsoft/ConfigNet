""" Tests for the script in the evaluation directory """
import pytest
import os
import sys
import glob

from fixtures import temporary_output_dir, test_asset_dir, model_dir, test_asset_dir_copy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import confignet

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "evaluation"))
import evaluate_confignet_controllability
import confignet_demo

@pytest.mark.parametrize("n_fine_tune_iters", [0, 1])
def test_confignet_evaluation(temporary_output_dir, model_dir, test_asset_dir, n_fine_tune_iters):
    model_path = os.path.join(model_dir, "confignet_256", "model.json")
    test_set_path = os.path.join(test_asset_dir, "test_dataset_res_256.pck")
    attribute_classifier_path = os.path.join(model_dir, "attribute_classifier", "model.json")

    script_arguments = "--model_path %s --test_set_path %s --output_dir %s"%(model_path, test_set_path, temporary_output_dir)
    script_arguments += " --attribute_classifier_path %s --n_samples 2"%attribute_classifier_path
    script_arguments += " --n_fine_tuning_iters %d --write_images"%n_fine_tune_iters

    evaluate_confignet_controllability.parse_args(script_arguments.split(" "))

    output_file_pattern = os.path.join(temporary_output_dir, "contr_metrics*")
    assert len(glob.glob(output_file_pattern)) > 0

@pytest.mark.parametrize("resolution", [256, 512])
def test_confignet_demo_latentgan_sampling(model_dir, resolution):
    base_script_arguments = "--resolution %s --test_mode"%resolution
    confignet_demo.run(base_script_arguments.split(" "))

@pytest.mark.parametrize("resolution", [256, 512])
def test_confignet_demo_multi_img(model_dir, test_asset_dir_copy, resolution):
    base_script_arguments = "--resolution %s --test_mode"%resolution
    script_arguments = base_script_arguments + " --image_path %s"%test_asset_dir_copy
    confignet_demo.run(script_arguments.split(" "))

@pytest.mark.parametrize("resolution", [256, 512])
def test_confignet_demo_single_img(model_dir, test_asset_dir_copy, resolution):
    base_script_arguments = "--resolution %s --test_mode"%resolution
    individual_image_path = os.path.join(test_asset_dir_copy, "img_0000000_000.png")
    script_arguments = base_script_arguments + " --image_path %s"%individual_image_path
    confignet_demo.run(script_arguments.split(" "))
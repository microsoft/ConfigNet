""" Tests of basic training and dataset creation scripts """
import sys
import os
import numpy as np
import cv2
import pytest

from fixtures import temporary_output_dir, test_asset_dir_copy, test_asset_dir

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import generate_dataset
from confignet import FaceImageNormalizer


# Ignoring this error since tensorflow throws it a lot
pytestmark = pytest.mark.filterwarnings("ignore:np.asscalar:DeprecationWarning")

def test_dataset_creation_script(temporary_output_dir, test_asset_dir_copy):
    script_arguments = "--dataset_dir %s --dataset_name %s"%(test_asset_dir_copy, "test_dataset")
    script_arguments += " --output_dir %s --img_output_dir %s"%(temporary_output_dir, temporary_output_dir)
    script_arguments += " --synthetic_data --load_attributes --img_size 256"

    generate_dataset.parse_args(script_arguments.split(" "))

    dataset_img_path = os.path.join(temporary_output_dir, "test_dataset_res_256_imgs.dat")
    dataset_pck_path = os.path.join(temporary_output_dir, "test_dataset_res_256.pck")

    assert os.path.exists(dataset_img_path)
    assert os.path.exists(dataset_pck_path)

def test_single_image_normalization(test_asset_dir):
    image_path = os.path.join(test_asset_dir, "img_0000000_000.png")
    image = cv2.imread(image_path)

    normalized_image = FaceImageNormalizer.normalize_individual_image(image, (256, 256))
    assert normalized_image.shape == (256, 256, 3)

    fake_image = np.zeros_like(image)
    normalized_image = FaceImageNormalizer.normalize_individual_image(fake_image, (256, 256))
    assert normalized_image is None
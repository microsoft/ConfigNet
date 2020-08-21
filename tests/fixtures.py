""" Fixtures shared by multiple test modules """
import pytest
import os
import sys
import tempfile
import tensorflow as tf
from distutils.dir_util import copy_tree

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from confignet import ConfigNet, CelebaAttributeClassifier
from confignet.metrics.celeba_attribute_prediction import DEFAULT_CONFIG as ATTR_CLASSIFIER_DEFAULT_CONFIG

@pytest.fixture()
def temporary_output_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture()
def test_asset_dir():
    return os.path.join(os.path.dirname(__file__), "test_assets")

@pytest.fixture()
def test_asset_dir_copy(test_asset_dir, temporary_output_dir):
    copy_tree(test_asset_dir, temporary_output_dir)
    return temporary_output_dir

@pytest.fixture()
def model_dir(test_asset_dir):
    return os.path.join(os.path.dirname(__file__), "..", "models")

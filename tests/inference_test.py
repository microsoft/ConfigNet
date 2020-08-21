import os
import sys
import shutil
import pytest
import cv2
import numpy as np
import tensorflow as tf

from fixtures import test_asset_dir, model_dir

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from confignet import FaceImageNormalizer, ConfigNet, LatentGAN


def get_normalized_test_image(test_asset_dir, output_shape):
    filename = "img_0000000_000.png"
    image_path = os.path.join(test_asset_dir, filename)
    image = cv2.imread(image_path)

    return FaceImageNormalizer.normalize_individual_image(image, output_shape)

@pytest.mark.parametrize("resolution", [256, 512])
def test_confignet_basic(test_asset_dir, model_dir, resolution):
    model_path = os.path.join(model_dir, "confignet_%d"%resolution, "model.json")
    model = ConfigNet.load(model_path)

    with tf.device('/cpu:0'):
        normalized_image = get_normalized_test_image(test_asset_dir, (resolution, resolution))
        embedding, rotation = model.encode_images(normalized_image[np.newaxis])
        decoded_image = model.generate_images(embedding, rotation)

        n_blendshapes = model.config["facemodel_inputs"]["blendshape_values"][0]

        neutral_expression = np.zeros((1, n_blendshapes), np.float32)
        modified_embedding = model.set_facemodel_param_in_latents(embedding, "blendshape_values", neutral_expression)
        decoded_image_modified = model.generate_images(embedding, rotation)

    reference_value_file = os.path.join(test_asset_dir, "confignet_basic_ref_%d.npz"%resolution)
    # set to True to save results as reference
    save_reference = False
    if save_reference:
        np.savez(reference_value_file, embedding=embedding, rotation=rotation,
                 decoded_image=decoded_image, modified_embedding=modified_embedding,
                 decoded_image_modified=decoded_image_modified)

    reference_vals = np.load(reference_value_file)
    assert np.allclose(embedding, reference_vals["embedding"])
    assert np.allclose(rotation, reference_vals["rotation"])
    assert np.allclose(decoded_image, reference_vals["decoded_image"])
    assert np.allclose(modified_embedding, reference_vals["modified_embedding"])
    assert np.allclose(decoded_image_modified, reference_vals["decoded_image_modified"])

@pytest.mark.parametrize("resolution", [256, 512])
def test_confignet_finetune(test_asset_dir, model_dir, resolution):
    model_path = os.path.join(model_dir, "confignet_%d"%resolution, "model.json")
    model = ConfigNet.load(model_path)

    normalized_image = get_normalized_test_image(test_asset_dir, (resolution, resolution))

    with tf.device('/cpu:0'):
        embedding, rotation = model.fine_tune_on_img(normalized_image[np.newaxis], n_iters=1)
        decoded_image = model.generate_images(embedding, rotation)

    reference_value_file = os.path.join(test_asset_dir, "confignet_finetune_ref_%d.npz"%resolution)
    # set to True to save results as reference
    save_reference = False
    if save_reference:
        np.savez(reference_value_file, embedding=embedding, rotation=rotation,
                 decoded_image=decoded_image)

    reference_vals = np.load(reference_value_file)
    assert np.allclose(embedding, reference_vals["embedding"])
    assert np.allclose(rotation, reference_vals["rotation"])
    assert np.allclose(decoded_image, reference_vals["decoded_image"])

@pytest.mark.parametrize("resolution", [256, 512])
def test_latent_gan(model_dir, test_asset_dir, resolution):
    latentgan_model_path = os.path.join(model_dir, "latentgan_%d"%resolution, "model.json")
    confignet_model_path = os.path.join(model_dir, "confignet_%d"%resolution, "model.json")

    latentgan = LatentGAN.load(latentgan_model_path)
    confignet = ConfigNet.load(confignet_model_path)

    np.random.seed(0)
    with tf.device('/cpu:0'):
        confignet_latents = latentgan.generate_latents(1)
        generated_imgs = confignet.generate_images(confignet_latents, np.zeros((1, 3)))

    reference_value_file = os.path.join(test_asset_dir, "latentgan_ref_%d.npz"%resolution)
    # set to True to save results as reference
    save_reference = False
    if save_reference:
        np.savez(reference_value_file, generated_imgs=generated_imgs)

    reference_vals = np.load(reference_value_file)
    assert np.allclose(generated_imgs, reference_vals["generated_imgs"])

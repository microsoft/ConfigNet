# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import sys
import argparse
import cv2
import numpy as np
import glob
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from confignet import ConfigNet, LatentGAN, FaceImageNormalizer
from confignet.confignet_utils import build_image_matrix
from basic_ui import BasicUI

def parse_args(args):
    model_base_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    confignet_model_paths = {
        256: os.path.join(model_base_dir, "confignet_256", "model.json"),
        512: os.path.join(model_base_dir, "confignet_512", "model.json")
    }
    latentgan_model_paths = {
        256: os.path.join(model_base_dir, "latentgan_256", "model.json"),
        512: os.path.join(model_base_dir, "latentgan_512", "model.json")
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="Path to either a directory of images or an individual image", default=None)
    parser.add_argument("--resolution", type=int, help="Path to ConfigNetModel", default=256)
    parser.add_argument("--n_rows", type=int, help="Number of rows in image matrix", default=2)
    parser.add_argument("--n_cols", type=int, help="Number of columns in image matrix", default=3)
    parser.add_argument("--test_mode", help="If specified only one frame will be generated, this is used in unit tests",
                        action="store_true", default=False)

    args = parser.parse_args(args)
    args.confignet_model_path = confignet_model_paths[args.resolution]
    args.latent_gan_model_path = latentgan_model_paths[args.resolution]

    return args

def process_images(image_path: str, resolution: int) -> List[np.ndarray]:
    '''Load the input images and normalize them'''
    if os.path.isfile(image_path):
        img = cv2.imread(image_path)
        img = FaceImageNormalizer.normalize_individual_image(img, (resolution, resolution))
        return [img]
    elif os.path.isdir(image_path):
        FaceImageNormalizer.normalize_dataset_dir(image_path, pre_normalize=True,
                                                  output_image_shape=(resolution, resolution), write_done_file=False)
        normalized_image_dir = os.path.join(image_path, "normalized")
        image_paths = glob.glob(os.path.join(normalized_image_dir, "*.png"))
        max_images = 200
        image_paths = image_paths[:max_images]
        if len(image_paths) == 0:
            raise ValueError("No images in input directory")
        imgs = []
        for path in image_paths:
            imgs.append(cv2.imread(path))
        return imgs
    else:
        raise ValueError("Image path is neither directory nor file")

def get_new_embeddings(args, input_images, latentgan_model: LatentGAN, confignet_model: ConfigNet):
    '''Samples new embeddings from either:
        - the LatentGAN if no input images were provided
        - by embedding the input images into the latent space using the real encoder
    '''
    if input_images is None:
        n_samples = args.n_rows * args.n_cols
        embeddings = latentgan_model.generate_latents(n_samples, truncation=0.7)
        rotations = np.zeros((n_samples, 3), dtype=np.float32)
        orig_images = confignet_model.generate_images(embeddings, rotations)
    else:
        # special case for one image so the demo is faster and nicer
        if len(input_images) == 1:
            args.n_rows = 1
            args.n_cols = 1
        n_samples = args.n_rows * args.n_cols
        sample_indices = np.random.randint(0, len(input_images), n_samples)
        orig_images = np.array([input_images[x] for x in sample_indices])
        embeddings, rotations = confignet_model.encode_images(orig_images)

    return embeddings, rotations, orig_images

def set_gaze_direction_in_embedding(latents: np.ndarray, eye_pose: np.ndarray, confignet_model: ConfigNet) -> np.ndarray:
    '''Sets the selected eye pose in the specified latent variables
       This is accomplished by passing the eye pose through the synthetic data encoder
       and setting the corresponding part of the latent vector.
    '''
    latents = confignet_model.set_facemodel_param_in_latents(latents, "bone_rotations:left_eye", eye_pose)
    return latents

def get_embedding_with_new_attribute_value(parameter_name:str, latents: np.ndarray, confignet_model: ConfigNet) -> np.ndarray:
    '''Samples a new value of the currently controlled face attribute and sets in the latent embedding'''

    new_param_value = confignet_model.facemodel_param_distributions[parameter_name].sample(1)[0]
    modified_latents = confignet_model.set_facemodel_param_in_latents(latents, parameter_name, new_param_value)

    return modified_latents

def print_intro():
    print("")
    print("This demo demonstrates the cabalities of ConfigNet to modify existing face images.")
    print("If no argument is specified the face will be ampled from a LatentGAN.")
    print("If you want to modify specific images pass a path to an individual image or a directory as argument.")
    print("")
    print("Simple things you might want to try:")
    print(" - use the WSAD keys to change the head pose of the faces you see")
    print(" - use the IKJL keys to change the eye geaze direction")
    print(" - press N to see how ConfigNet can vary illumination")
    print(" - press X to sample a new value of a chosen attribute")
    print(" - use Z and C to change the attribute that is sampled using X")

    print("")
    print("")

def print_instructions():
    print("Here are the keys you can use to control the demo.")
    print("Make sure that the window with the face renders is in focus when pressing the keys.")
    print("")
    print("space - sample another set of images from input directory or latent gan (if input directory not specified")
    BasicUI.print_instructions()
    print("X - sample new value of currently controlled face model parameter")
    print("V - reset latent embedding back to original")
    print("B - fine-tune the generator on the chosen image (one-shot learning), works only if a single image is passed as input")
    print("H - see this message")

    print("")
    print("")

def run(args):
    print_intro()
    print_instructions()

    args = parse_args(args)
    if args.image_path is not None:
        input_images = process_images(args.image_path, args.resolution)
        latentgan_model = None
    else:
        input_images = None
        print("WARNING: no input image directory specified, embeddings will be sampled using Laten GAN")
        latentgan_model = LatentGAN.load(args.latent_gan_model_path)
    confignet_model = ConfigNet.load(args.confignet_model_path)

    basic_ui = BasicUI(confignet_model)

    # Sample latent embeddings from input images if available and if not sample from Latent GAN
    current_embedding_unmodified, current_rotation, orig_images = get_new_embeddings(args, input_images, latentgan_model, confignet_model)
    # Set next embedding value for rendering
    basic_ui.set_next_embeddings(current_embedding_unmodified)


    while not basic_ui.exit:
        # This interpolates between the previous and next set embeddings
        current_renderer_input = basic_ui.get_current_frame_embeddings()
        # Set eye gaze direction as controlled by the user
        current_renderer_input = set_gaze_direction_in_embedding(current_renderer_input, basic_ui.eye_rotation_offset, confignet_model)

        generated_imgs = confignet_model.generate_images(current_renderer_input, current_rotation + basic_ui.rotation_offset)

        white_strip = np.full((generated_imgs.shape[0], generated_imgs.shape[1], 20, 3), 255, np.uint8)
        visualization_imgs = np.dstack((orig_images, generated_imgs, white_strip))

        image_matrix = build_image_matrix(visualization_imgs, args.n_rows, args.n_cols)

        basic_ui.perform_per_frame_actions()

        if not args.test_mode:
            key = cv2.imshow("img", image_matrix)
        key = cv2.waitKey(1)

        key = basic_ui.drive_ui(key, args.test_mode)

        if key == ord(" ") or args.test_mode:
            current_embedding_unmodified, current_rotation, orig_images = get_new_embeddings(args, input_images, latentgan_model, confignet_model)
            basic_ui.set_next_embeddings(current_embedding_unmodified)
        if key == ord("v") or args.test_mode:
            basic_ui.set_next_embeddings(current_embedding_unmodified)
        if key == ord("x") or args.test_mode:
            current_attribute_name = basic_ui.facemodel_param_names[basic_ui.controlled_param_idx]
            new_embedding_value = get_embedding_with_new_attribute_value(current_attribute_name, basic_ui.get_current_frame_embeddings(), confignet_model)
            basic_ui.set_next_embeddings(new_embedding_value)
        if key == ord("b") or args.test_mode:
            if input_images is None or len(input_images) != 1:
                print("For one-shot learning to work you need to specify a single input image path")
                continue
            if args.test_mode:
                n_fine_tuning_iters = 1
            else:
                n_fine_tuning_iters = 50
            print("Fine tuning generator on single image, this might take a minute or two")
            current_embedding_unmodified, current_rotation = confignet_model.fine_tune_on_img(input_images[0], n_fine_tuning_iters)
            basic_ui.set_next_embeddings(current_embedding_unmodified)
        if key == ord("h") or args.test_mode:
            print_intro()
            basic_ui.print_instructions()
            print_instructions()

        if args.test_mode:
            break

if __name__ == "__main__":
    run(sys.argv[1:])
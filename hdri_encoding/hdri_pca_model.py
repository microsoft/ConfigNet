# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Class and utilities for building a PCA model of 360 hdri images"""

import os
import sys
import glob
import pickle
import numpy as np
import cv2
from sklearn.decomposition import PCA

import argparse

class HDRIModelPCA:
    def __init__(self, output_shape, n_rotations_per_image):
        self.n_rotations_per_image = n_rotations_per_image
        self.output_shape = output_shape

        self.pca_model = None

    # n_components has same meaning as in sklearn's PCA
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    def fit(self, hdri_images, n_components=0.9):
        hdri_images = np.log2(hdri_images + 1)

        hdri_images_rot = apply_random_rotations(hdri_images, self.n_rotations_per_image)
        hdri_images_rot = resize_hdris(hdri_images_rot, self.output_shape)

        hdri_images_rot = np.reshape(hdri_images_rot, (hdri_images_rot.shape[0], -1))

        if n_components > 1:
            n_components = int(n_components)
        self.pca_model = PCA(n_components, svd_solver="full", whiten=True)
        self.pca_model.fit(hdri_images_rot)

        explained_variance = np.sum(self.pca_model.explained_variance_ratio_)
        print("PCA model fitted, %0.2f%% of variance explained by %d components"%(100 * explained_variance, self.pca_model.components_.shape[0]))

    def write_basis_images(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, basis in enumerate(self.pca_model.components_):
            reshaped_basis = np.reshape(basis, (*self.output_shape, 3))
            reshaped_basis = 255 * (reshaped_basis - np.min(reshaped_basis)) / (np.max(reshaped_basis) - np.min(reshaped_basis))
            reshaped_basis = reshaped_basis.astype(np.uint8)

            cv2.imwrite(os.path.join(output_dir, str(i).zfill(3) + ".png"), reshaped_basis)

    def transform(self, hdri_images, rotations=None):
        hdri_images = np.log2(hdri_images + 1)

        if rotations is not None:
            assert len(rotations) == len(hdri_images)
            for i in range(len(hdri_images)):
                hdri_images[i] = rotate_hdri(hdri_images[i], rotations[i])

        hdri_images = resize_hdris(hdri_images, self.output_shape)
        hdri_images = np.reshape(hdri_images, (hdri_images.shape[0], -1))

        transformed = self.pca_model.transform(hdri_images)

        return transformed

    def inverse_transform(self, X):
        hdri_images = self.pca_model.inverse_transform(X)
        hdri_images = np.reshape(hdri_images, (len(hdri_images), *self.output_shape, 3))
        hdri_images = np.power(2, hdri_images) - 1

        return hdri_images

    def save(self, output_path):
        pickle.dump(self, open(output_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(input_path):
        return pickle.load(open(input_path, "rb"))

def load_hdris(hdri_dir):
    hdri_paths = glob.glob(os.path.join(hdri_dir, "*.hdr"))

    hdri_images = []
    for hdri_path in hdri_paths:
        image = cv2.imread(hdri_path, -1)
        hdri_images.append(image)

    return np.array(hdri_images), hdri_paths

def apply_random_rotations(hdri_images, rotations_per_image):
    rotated_hdri_images = np.zeros((hdri_images.shape[0] * rotations_per_image, *hdri_images.shape[1:]), dtype=hdri_images.dtype)

    i = 0
    for image in hdri_images:
        for _ in range(rotations_per_image):
            rotation = np.random.uniform(0, 360)
            rotated_image = rotate_hdri(image, rotation)
            rotated_hdri_images[i] = rotated_image
            i += 1

    return rotated_hdri_images

def resize_hdris(hdri_images, output_shape):
    resized_hdri_images = []

    for image in hdri_images:
        resized = cv2.resize(image, output_shape[::-1], interpolation=cv2.INTER_AREA)
        resized_hdri_images.append(resized)

    return np.array(resized_hdri_images, dtype=hdri_images.dtype)

def rotate_hdri(hdri_image, rotation_deg):
    n_cols = hdri_image.shape[1]
    shift = int(round(rotation_deg * n_cols / 360))

    return np.roll(hdri_image, shift, axis=1)

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdri_dir", help="Directory containing the .hdr images", required=True)
    parser.add_argument("--output_dir", help="Directory to which the produced model and the optional outputs will be written", required=True)
    parser.add_argument("--n_components", type=float, help="Number of PCA components, if < 1 denotes the fraction of variance explained", default=50)
    parser.add_argument("--output_shape", type=int, nargs=2, help="Shape to which .hdr images will be scaled for training: height, width", default=(64, 128))
    parser.add_argument("--n_rotations_per_image", type=int, help="Number of random rotations applied to each .hdr image", default=5)
    parser.add_argument("--write_hdris", help="If specified the reconstructed HDRIs will be writtern", default=False, action="store_true")
    parser.add_argument("--seed", type=int, help="Seed for random number generator", default=0)
    args = parser.parse_args(args)
    args.output_shape = tuple(args.output_shape)

    np.random.seed(args.seed)

    hdri_images, _ = load_hdris(args.hdri_dir)
    print("HDRIs loaded")
    pca_model = HDRIModelPCA(args.output_shape, args.n_rotations_per_image)
    pca_model.fit(hdri_images, args.n_components)

    pca_model.save(os.path.join(args.output_dir, "hdri_model.pck"))

    pca_model.write_basis_images(os.path.join(args.output_dir, "pca_basis"))

    if args.write_hdris:
        hdri_output_dir = os.path.join(args.output_dir, "hdris")
        if not os.path.exists(hdri_output_dir):
            os.makedirs(hdri_output_dir)
        hdris_in_pca_space = pca_model.transform(hdri_images)
        hdri_images_reconstructed = pca_model.inverse_transform(hdris_in_pca_space)
        for i, image in enumerate(hdri_images_reconstructed):
            cv2.imwrite(os.path.join(hdri_output_dir, str(i).zfill(3) + "_reconstructed.hdr"), image)

        hdri_images = resize_hdris(hdri_images, args.output_shape)
        for i, image in enumerate(hdri_images):
            cv2.imwrite(os.path.join(hdri_output_dir, str(i).zfill(3) + "_original.hdr"), image)

if __name__ == "__main__":
    parse_args(sys.argv[1:])
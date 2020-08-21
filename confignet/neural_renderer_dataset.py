# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Class for storing training data for neural renderer training"""

import numpy as np
import os
import sys
import cv2
import glob
import pickle
import ntpath
import json
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
import transformations
from typing import List, Dict, Any, Tuple

from . import dataset_utils
from .face_image_normalizer import FaceImageNormalizer
from .metrics import inception_distance

class OneHotDistribution:
    """A uniform discrete distribution represented as one-hot vector

    Has same interface as scikitlearn's GMM.
    """

    def __init__(self):
        self.n_features = None

    def fit(self, X):
        self.n_features = X.shape[1]

    def sample(self, n_samples=1):
        sampled_indices = np.random.randint(0, self.n_features, size=n_samples)
        one_hot = np.zeros((n_samples, self.n_features), np.float32)
        one_hot[np.arange(n_samples), sampled_indices] = 1

        return one_hot, sampled_indices

class ExemplarDistribution:
    """An arbitrary exemplar-based distribution

    Has same interface as scikitlearn's GMM.
    """

    def __init__(self):
        self.examplars = None
        self.n_exemplars = None

    def fit(self, X):
        self.exemplars = X
        self.n_exemplars = self.exemplars.shape[0]

    def sample(self, n_samples=1):
        sampled_indices = np.random.randint(0, self.n_exemplars, size=n_samples)
        output = self.exemplars[sampled_indices]

        return output, None

class EyeRegionSpec:
    """Specs of the eye region in the UV space of the 3D model used in synthetic data"""
    eye_region_max_y = 0.15
    eye_region_min_y = 0.07

    l_eye_region_max_x = 0.16
    l_eye_region_min_x = 0.09
    r_eye_region_max_x = 0.91
    r_eye_region_min_x = 0.84

class NeuralRendererDataset:
    """Dataset used in the training of CONFIG and other DNNs in this repo"""
    def __init__(self, img_shape: Tuple[int, int, int], is_synthetic: bool,
                 head_rotation_range=((-30, 30), (-10, 10), (0, 0)), eye_rotation_range=((-25, 25), (-15, 15), (0, 0))):
        self.img_shape = img_shape
        self.is_synthetic = is_synthetic
        self.head_rotation_range = np.array(head_rotation_range)
        self.eye_rotation_range = np.array(eye_rotation_range)

        # These things are generated at dataset creation
        self.imgs = None
        self.imgs_memmap_filename = None
        self.imgs_memmap_shape = None
        self.imgs_memmap_dtype = None

        self.inception_features = None
        self.render_metadata = None

        # Masks showing location of eye region in the synthetic data
        self.eye_masks = None
        # CelebA attributes
        self.attributes = None


        # These things are generated at training time
        # Processed render_metadata, computed based on DNN config
        self.metadata_inputs = None
        self.metadata_input_distributions = None
        # Labels for each metadat input, for example names of expressions that correspond to various blendshapes
        self.metadata_input_labels = None

    def generate_face_dataset(self, input_dir: str, output_path: str, attribute_label_file_path=None, pre_normalize=True) -> None:
        FaceImageNormalizer.normalize_dataset_dir(input_dir, pre_normalize, self.img_shape)

        image_paths = []
        image_paths.extend(glob.glob(os.path.join(input_dir, "normalized", "*.png")))

        if self.is_synthetic:
            metadata = self._load_metadata(image_paths)
            image_paths, metadata = self._remove_samples_with_out_of_range_pose(image_paths, metadata)
            self.render_metadata = metadata
            self.eye_masks = []

        if attribute_label_file_path is not None:
            image_attributes = dataset_utils.parse_celeba_attribute_file(attribute_label_file_path)
            self.attributes = []

        self._initialize_imgs_memmap(len(image_paths), output_path)

        for i in range(len(image_paths)):
            if i % max([1, (len(image_paths) // 100)]) == 0:
                perc_complete = 100 * i / len(image_paths)
                print("Image reading %d%% complete"%(perc_complete))

            img_filename_with_ext = ntpath.basename(image_paths[i])
            img_filename = img_filename_with_ext.split('.')[0]

            if self.attributes is not None:
                self.attributes.append(image_attributes[img_filename])

            self.imgs[i] = cv2.imread(image_paths[i])
            if self.is_synthetic:
                eye_mask = self._get_eye_mask_for_image_path(image_paths[i])
                self.eye_masks.append(eye_mask)

        if self.is_synthetic:
            self.eye_masks = np.array(self.eye_masks)

        self._compute_inception_features()
        self.save(output_path)

    def _initialize_imgs_memmap(self, n_images: int, output_path: str) -> None:
        self.imgs_memmap_shape = (n_images, *self.img_shape)
        self.imgs_memmap_dtype = np.uint8
        self.imgs_memmap_filename = os.path.splitext(os.path.basename(output_path))[0] + "_imgs.dat"
        basedir = os.path.dirname(output_path)
        self.imgs = np.memmap(os.path.join(basedir, self.imgs_memmap_filename), self.imgs_memmap_dtype, "w+", shape=self.imgs_memmap_shape)


    def process_metadata(self, config: Dict[str, Any], update_config=False) -> None:
        """Preprocesses self.render_metadata to a format that can be ingested in CONFIG training based on the network config.

        Reads metadata inputs that are to be used from the config.
        If update_config is specified, the method updates the input dimensionality in the config.
        Updates dictionary of metadata input vectors, each vector corresponds to a different metadata type (hair style, texture, etc).
        """

        self.metadata_inputs = {}
        self.metadata_input_distributions = {}
        self.metadata_input_labels = {}

        def fit_distribution(data, distr_type):
            if distr_type == "GMM":
                distr = GaussianMixture()
                distr.fit(data)
            elif distr_type == "one_hot":
                distr = OneHotDistribution()
                distr.fit(data)
            elif distr_type == "exemplar":
                distr = ExemplarDistribution()
                distr.fit(data)

            return distr

        for input_name in config["facemodel_inputs"].keys():
            nested_dict_keys = input_name.split(":")
            metadata_values = self.render_metadata
            for key in nested_dict_keys:
                metadata_values = [metadata[key] for metadata in metadata_values]

            # Some string values can be set to none, we'll replace that with a string for consistency
            for i, metadata_value in enumerate(metadata_values):
                if metadata_value is None:
                    metadata_values[i] = "none"

            assert all([type(metadata_value) == type(metadata_values[0]) for metadata_value in metadata_values])

            if type(metadata_values[0]) == str:
                unique_vals, unique_inverse = np.unique(metadata_values, return_inverse=True)
                n_unique_vals = unique_vals.shape[0]
                one_hot = np.zeros((len(metadata_values), n_unique_vals))
                one_hot[np.arange(len(metadata_values)), unique_inverse] = 1
                self.metadata_inputs[input_name] = one_hot
                self.metadata_input_distributions[input_name] = fit_distribution(one_hot, "one_hot")
                self.metadata_input_labels[input_name] = unique_vals.tolist()
                if update_config:
                    config["facemodel_inputs"][input_name] = (int(n_unique_vals), config["facemodel_inputs"][input_name][1])
            elif type(metadata_values[0]) == list:
                assert all([len(value) == len(metadata_values[0]) for value in metadata_values])

                metadata_values = np.array(metadata_values, dtype=np.float32)
                self.metadata_inputs[input_name] = metadata_values
                self.metadata_input_distributions[input_name] = fit_distribution(metadata_values, "exemplar")
                self.metadata_input_labels[input_name] = None
                if update_config:
                    config["facemodel_inputs"][input_name] = (metadata_values.shape[1], config["facemodel_inputs"][input_name][1])
            elif type(metadata_values[0]) == dict:
                assert all([metadata_value.keys() == metadata_values[0].keys() for metadata_value in metadata_values])

                metadata_values = [OrderedDict(sorted(metadata_value.items(), key=lambda t: t[0])) for metadata_value in metadata_values]

                self.metadata_input_labels[input_name] = list(metadata_values[0].keys())
                metadata_values = np.array([list(metadata_value.values()) for metadata_value in metadata_values], dtype=np.float32)
                if input_name == "blendshape_values":
                    jaw_opening_values = np.array([metadata["bone_rotations"]["jaw"][0] for metadata in self.render_metadata])
                    metadata_values = np.hstack((metadata_values, jaw_opening_values[:, np.newaxis]))
                    self.metadata_input_labels[input_name].append("jaw_opening")

                self.metadata_inputs[input_name] = metadata_values
                self.metadata_input_distributions[input_name] = fit_distribution(metadata_values, "exemplar")
                if update_config:
                    config["facemodel_inputs"][input_name] = (metadata_values.shape[1], config["facemodel_inputs"][input_name][1])

        # Extract rotation angles
        rotation_values = [metadata["bone_rotations"]["head"] for metadata in self.render_metadata]
        self.metadata_inputs["rotations"] = np.array(rotation_values)[:, [2, 0, 1]]
        # No label is specified for vector inputs
        self.metadata_input_labels["rotations"] = None

    def _load_metadata(self, image_paths: List[str]) -> Dict[str, Any]:
        image_paths_split = [os.path.split(os.path.splitext(path)[0]) for path in image_paths]
        metadata_paths = [os.path.join(head_tail[0], "..", "meta" + head_tail[1][3:] + ".json") for head_tail in image_paths_split]

        render_metadata = []
        for path in metadata_paths:
            metadata = json.load(open(path))
            render_metadata.append(metadata)

        return render_metadata

    def _get_eye_mask_for_image_path(self, image_path: str) -> np.ndarray:
        image_path_split = os.path.split(os.path.splitext(image_path)[0])
        uv_map_path = os.path.join(image_path_split[0], "uv" + image_path_split[1][3:] + ".exr")

        uv_img = cv2.imread(uv_map_path, -1)
        l_eye_pixel_idxs = np.where((uv_img[:, :, 0] < EyeRegionSpec.l_eye_region_max_x) & (uv_img[:, :, 0] > EyeRegionSpec.l_eye_region_min_x) &
                                (uv_img[:, :, 1] < EyeRegionSpec.eye_region_max_y) & (uv_img[:, :, 1] > EyeRegionSpec.eye_region_min_y))
        r_eye_pixel_idxs = np.where((uv_img[:, :, 0] < EyeRegionSpec.r_eye_region_max_x) & (uv_img[:, :, 0] > EyeRegionSpec.r_eye_region_min_x) &
                                (uv_img[:, :, 1] < EyeRegionSpec.eye_region_max_y) & (uv_img[:, :, 1] > EyeRegionSpec.eye_region_min_y))

        eye_mask = np.zeros(uv_img.shape[:2], dtype=np.uint8)
        eye_mask[l_eye_pixel_idxs] = 1
        eye_mask[r_eye_pixel_idxs] = 1

        return eye_mask

    def _remove_samples_with_out_of_range_pose(self, image_paths: List[str], metadata: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        # Change axis order to the one used in ConfigNet (y, x, z) from the one used in synthetics
        head_rotation_range_for_synth = np.pi * self.head_rotation_range[[1, 2, 0]] / 180
        eye_rotation_range_for_synth = np.pi * self.eye_rotation_range[[1, 2, 0]] / 180

        rejected_image_idxs = []
        for i, image_metadata in enumerate(metadata):
            head_pose = image_metadata["bone_rotations"]["head"]
            eye_pose = image_metadata["bone_rotations"]["left_eye"]

            head_lower_bound = np.all(head_pose >= head_rotation_range_for_synth[:, 0])
            head_upper_bound = np.all(head_pose <= head_rotation_range_for_synth[:, 1])

            eye_lower_bound = np.all(eye_pose >= eye_rotation_range_for_synth[:, 0])
            eye_upper_bound = np.all(eye_pose <= eye_rotation_range_for_synth[:, 1])

            if not (head_lower_bound and head_upper_bound and eye_lower_bound and eye_upper_bound):
                rejected_image_idxs.append(i)

        image_paths = [image_path for i, image_path in enumerate(image_paths) if i not in rejected_image_idxs]
        metadata = [image_metadata for i, image_metadata in enumerate(metadata) if i not in rejected_image_idxs]

        return image_paths, metadata

    def write_images(self, directory: str, draw_landmarks=False) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)

        for i in range(len(self.imgs)):
            if draw_landmarks:
                img = np.copy(self.imgs[i])
                for landmark in self.landmarks[i]:
                    cv2.circle(img, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0))
            else:
                img = self.imgs[i]

            cv2.imwrite(os.path.join(directory, str(i).zfill(5) + ".jpg"), img)

        mean_img = np.mean(self.imgs, axis=0).astype(np.uint8)
        cv2.imwrite(os.path.join(directory, "mean_img.jpg"), mean_img)

    def write_images_by_attribute(self, directory: str) -> None:
        assert self.attributes is not None
        assert all([self.attributes[0].keys() == img_attributes.keys() for img_attributes in self.attributes])

        attribute_names = self.attributes[0].keys()
        for attribute_name in attribute_names:
            imgs_with_attribute = [i for i, img_attributes in enumerate(self.attributes) if img_attributes[attribute_name]]

            attribute_output_dir = os.path.join(directory, attribute_name)
            if not os.path.exists(attribute_output_dir):
                os.makedirs(attribute_output_dir)
            for img_idx in imgs_with_attribute:
                cv2.imwrite(os.path.join(attribute_output_dir, str(img_idx).zfill(6) + ".jpg"), self.imgs[img_idx])

    def get_attribute_values(self, sample_idxs: List[int], attribute_names: List[str]) -> np.ndarray:
        assert self.attributes is not None

        attribute_values = []
        for idx in sample_idxs:
            sample_attributes = self.attributes[idx]
            attribute_present = [sample_attributes[attribute_name] for attribute_name in attribute_names]
            attribute_values.append(attribute_present)

        return np.array(attribute_values)

    def _compute_inception_features(self) -> None:
        feature_extractor = inception_distance.InceptionFeatureExtractor(self.imgs.shape[1:])
        self.inception_features = feature_extractor.get_features(self.imgs)

    def save(self, filename: str) -> None:
        # Delete the memory-mapped image array so it does not get pickled
        del self.imgs
        self.imgs = None
        pickle.dump(self, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        # Re-load the image array
        basedir = os.path.dirname(filename)
        self.imgs = np.memmap(os.path.join(basedir, self.imgs_memmap_filename), self.imgs_memmap_dtype, "r", shape=self.imgs_memmap_shape)

    @staticmethod
    def load(filename: str) -> "NeuralRendereDataset":
        # Older datasets might not load properly due to changes in repo structure, the code in except fixes it
        try:
            dataset = pickle.load(open(filename, "rb"))
        except:
            from . import neural_renderer_dataset
            sys.modules["neural_renderer_dataset"] = neural_renderer_dataset
            dataset = pickle.load(open(filename, "rb"))

        basedir = os.path.dirname(filename)
        dataset.imgs = np.memmap(os.path.join(basedir, dataset.imgs_memmap_filename), dataset.imgs_memmap_dtype, "r", shape=dataset.imgs_memmap_shape)

        return dataset

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import numpy as np
import glob
import cv2
import transformations
import tempfile

from typing import Tuple

from . import dataset_utils

repo_root_dir_path = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_OPENFACE_PATH = os.path.join(repo_root_dir_path, "3rd_party", "OpenFace_2.2.0_win_x64", "FaceLandmarkImg.exe")

class FaceImageNormalizer:
    """Responsible for normalizing face images to the format used by ConfigNet

    There are two steps to this process:
     - pre-normalization (not used for synthetic data where the faces are centralized already)
     - 3D head-center normalization
    The pre-normalization puts the head in the center of the image.
    This step is necessary as the 3D head center normalization process does not work correctly on non-centered faces.
    Head-center normalization estimates the projection of the center of the head and ensures that this point is in the center of the output image.
    """

    # Constants for pre-normalization
    ref_pre_norm_landmark_idxs = ((36, 39), (42, 45), (30,), (48,), (54,))
    ref_pre_norm_landmark_positions = np.array(((0.32, 0.45), (0.68, 0.45), (0.5, 0.6), (0.34, 0.82), (0.66, 0.82)))
    # Face_scale controlls the size of the face in the pre-normalizedimage, 1.0 means face takes most of the image
    pre_norm_face_scale = 0.5
    pre_norm_image_size = 1024
    ref_pre_norm_landmark_positions = (ref_pre_norm_landmark_positions - 0.5) * pre_norm_face_scale + 0.5

    # Constants for head-center normalization
    ref_head_center_coords = (0.5, 0.42),
    eye_corner_idxs = (36, 45)
    mouth_top_idx = (51)
    head_center_idxs = (0, 16) # the 3D landmarks in the middle of which the head center lies
    interocular_fraction = 0.45
    eye_to_mouth_fraction = 0.34

    image_filename_patterns = ("*.jpg", "*.png", "*.bmp", "*.jpeg")

    @classmethod
    def normalize_dataset_dir(cls, input_dir: str, pre_normalize:bool, output_image_shape:Tuple[int, int],
                              openface_path=DEFAULT_OPENFACE_PATH, write_done_file=True) -> None:
        output_dir = os.path.join(input_dir, "normalized")
        done_file_path = os.path.join(output_dir, "normalization_done")
        if os.path.exists(done_file_path):
            return

        dataset_utils.run_openface_on_dir(input_dir, openface_path)
        if pre_normalize:
            size = cls.pre_norm_image_size
            pre_norm_dir = os.path.join(input_dir, "pre_normalized")
            pre_norm_done_file_path = os.path.join(pre_norm_dir, "normalization_done")
            if not os.path.exists(pre_norm_done_file_path):
                cls._normalize_directory(input_dir, pre_norm_dir, True, (size, size))
                dataset_utils.run_openface_on_dir(pre_norm_dir, openface_path)
                if write_done_file:
                    with open(pre_norm_done_file_path, "w+") as _:
                        pass
            input_dir = pre_norm_dir

        cls._normalize_directory(input_dir, output_dir, False, output_image_shape)
        if write_done_file:
            with open(done_file_path, "w+") as _:
                pass

    @classmethod
    def normalize_individual_image(cls, image: np.ndarray, output_image_shape:Tuple[int, int]) -> np.ndarray:
        with tempfile.TemporaryDirectory() as temp_dir:
            cv2.imwrite(os.path.join(temp_dir, "temp_img.png"), image)
            cls.normalize_dataset_dir(temp_dir, True, output_image_shape)
            normalized_image_path = os.path.join(temp_dir, "normalized", "temp_img.png")
            if os.path.exists(normalized_image_path):
                normalized_image = cv2.imread(normalized_image_path)
            else:
                normalized_image = None

        return normalized_image

    @classmethod
    def _normalize_directory(cls, input_dir:str, output_dir:str, normalize_2D:bool, output_image_shape:Tuple[int, int]) -> None:
        os.makedirs(output_dir, exist_ok=True)

        image_paths = []
        for pattern in cls.image_filename_patterns:
            image_paths.extend(glob.glob(os.path.join(input_dir, pattern)))

        for image_path in image_paths:
            img_filename_with_ext = os.path.basename(image_path)
            img_filename = os.path.splitext(img_filename_with_ext)[0]

            csv_path = os.path.join(input_dir, "processed", img_filename + ".csv")

            if not os.path.exists(csv_path):
                continue

            landmarks, landmarks_3d, pose = dataset_utils.read_landmarks_and_pose_from_csv(csv_path)
            if landmarks is None:
                continue
            details_file_path = os.path.join(input_dir, "processed", img_filename + "_of_details.txt")
            estimated_intrinsics = dataset_utils.read_estimated_intrinsics(details_file_path)

            if normalize_2D:
                M = cls._get_normalizing_transform_2d(landmarks, output_image_shape)
            else:
                M = cls._get_normalizing_transform_3d(landmarks, landmarks_3d, pose, estimated_intrinsics, output_image_shape)

            image = cv2.imread(image_path)
            image = cv2.warpAffine(image, M, output_image_shape[:2])
            cv2.imwrite(os.path.join(output_dir, img_filename + ".png"), image)


            image_path_split = os.path.split(os.path.splitext(image_path)[0])
            uv_image_name = "uv" + image_path_split[1][3:] + ".exr"
            uv_image_path = os.path.join(input_dir, uv_image_name)
            if os.path.exists(uv_image_path):
                uv_image = cv2.imread(uv_image_path, -1)
                # INTER_NEAREAST as interpolation in UV space has unpredictable results
                uv_image = cv2.warpAffine(uv_image, M, output_image_shape[:2], flags=cv2.INTER_NEAREST)

                cv2.imwrite(os.path.join(output_dir, uv_image_name), uv_image)

    @classmethod
    def _get_normalizing_transform_3d(cls, landmarks_2d: np.ndarray, landmarks_3d: np.ndarray,
                                      pose: np.ndarray, intrinsics: np.ndarray, output_image_shape: Tuple[int, int]) -> np.ndarray:
        ref_interocular_distance = cls.interocular_fraction * output_image_shape[1]
        ref_eye_to_mouth_distance = cls.eye_to_mouth_fraction * output_image_shape[0]

        t = pose[:3]
        R = transformations.euler_matrix(pose[3], pose[4], pose[5], axes="rxyz")[:3, :3]

        landmarks_3d_canonical_pose = np.dot((landmarks_3d - t), R)
        landmarks_3d_frontal = landmarks_3d_canonical_pose + t

        frontal_landmarks_proj = np.dot(landmarks_3d_frontal, intrinsics.T)
        frontal_landmarks_proj = frontal_landmarks_proj[:, :2] / frontal_landmarks_proj[:, [2]]

        frontal_interocular_distance = np.linalg.norm(frontal_landmarks_proj[cls.eye_corner_idxs[0]] - frontal_landmarks_proj[cls.eye_corner_idxs[1]])
        frontal_eye_center = (frontal_landmarks_proj[cls.eye_corner_idxs[0]] + frontal_landmarks_proj[cls.eye_corner_idxs[1]]) / 2
        frontal_mouth_to_eye_distance = np.linalg.norm(frontal_landmarks_proj[cls.mouth_top_idx] - frontal_eye_center)
        image_scale_interocular = ref_interocular_distance / frontal_interocular_distance
        image_scale_eye_to_mouth = ref_eye_to_mouth_distance / frontal_mouth_to_eye_distance
        image_scale = (image_scale_interocular + image_scale_eye_to_mouth) / 2
        # Image rotation
        eye_to_eye_vector = landmarks_2d[cls.eye_corner_idxs[1]] - landmarks_2d[cls.eye_corner_idxs[0]]
        image_rotation = np.arctan2(eye_to_eye_vector[1], eye_to_eye_vector[0])

        # Image translation
        head_center_coords = np.mean(landmarks_3d[cls.head_center_idxs, :], axis=0)
        head_center_proj = np.dot(head_center_coords, intrinsics.T)
        head_center_proj = head_center_proj[:2] / head_center_proj[2]

        sin_rot = np.sin(image_rotation)
        cos_rot = np.cos(image_rotation)
        output_a = image_scale * np.array(((cos_rot, sin_rot), (-sin_rot, cos_rot)))

        output_t = cls.ref_head_center_coords * np.array(output_image_shape[:2]) - np.dot(output_a, head_center_proj)

        M = np.hstack((output_a, output_t.T))
        return M

    @classmethod
    def _get_normalizing_transform_2d(cls, landmarks: np.ndarray, output_image_shape: Tuple[int, int]) -> np.ndarray:
        incoming_ref_landmarks = np.array([np.mean(landmarks[idxs, :], axis=0) for idxs in cls.ref_pre_norm_landmark_idxs])
        ref_landmark_positions = cls.ref_pre_norm_landmark_positions * np.array(output_image_shape[:2])
        A, t = dataset_utils.get_similarity_transform(ref_landmark_positions, incoming_ref_landmarks)

        M = np.hstack((A, t[:, np.newaxis]))
        return M
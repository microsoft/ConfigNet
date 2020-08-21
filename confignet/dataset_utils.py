# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
import os
import cv2
import numpy as np
import subprocess

def read_landmarks_and_pose_from_csv(csv_file_path, n_landmarks=68, confidence_threshold=0.6):
    landmark_column_idxs, landmark_3d_column_idxs = get_landmark_column_idxs_from_csv(csv_file_path, n_landmarks)
    pose_column_idxs = get_pose_column_idxs_from_csv(csv_file_path)
    landmarks = np.loadtxt(csv_file_path, skiprows=1, usecols=landmark_column_idxs, delimiter=',')
    landmarks_3d = np.loadtxt(csv_file_path, skiprows=1, usecols=landmark_3d_column_idxs, delimiter=',')
    pose = np.loadtxt(csv_file_path, skiprows=1, usecols=pose_column_idxs, delimiter=',')

    file = open(csv_file_path, "r")
    headers = file.readline().split(",")
    file.close()
    #Remove leading and trailing spaces from header names
    headers = [header.strip() for header in headers]
    confidence_header = "confidence"
    confidence_header_idx = headers.index(confidence_header)
    confidences = np.loadtxt(csv_file_path, skiprows=1, usecols=[confidence_header_idx], delimiter=',')

    #If more than 1 face detected
    if len(landmarks.shape) > 1:
        most_confident_face_idx = np.argmax(confidences)
        landmarks = landmarks[most_confident_face_idx]
        landmarks_3d = landmarks_3d[most_confident_face_idx]
        pose = pose[most_confident_face_idx]
        confidence = confidences[most_confident_face_idx]
    else:
        confidence = confidences

    if confidence < confidence_threshold:
        return None, None, None

    landmarks = np.reshape(landmarks, (n_landmarks, 2), order='F')
    landmarks_3d = np.reshape(landmarks_3d, (n_landmarks, 3), order='F')

    return landmarks, landmarks_3d, pose

def read_estimated_intrinsics(details_file_path):
    with open(details_file_path, "r") as fp:
        lines = fp.readlines()

    camera_params_line = lines[2]
    camera_params = camera_params_line.split(":")[1]
    camera_params = camera_params.split(",")
    camera_params = [float(x) for x in camera_params]

    K = np.eye(3)
    K[0, 0] = camera_params[0]
    K[1, 1] = camera_params[1]
    K[0, 2] = camera_params[2]
    K[1, 2] = camera_params[3]

    return K

def get_landmark_column_idxs_from_csv(csv_file_path, n_landmarks):
    file = open(csv_file_path, "r")
    headers = file.readline().split(",")
    file.close()

    #Remove leading and trailing spaces from header names
    headers = [header.strip() for header in headers]

    landmarks_headers = ["x_" + str(i) for i in range(n_landmarks)]
    landmarks_headers.extend(["y_" + str(i) for i in range(n_landmarks)])
    landmark_column_idxs = [headers.index(landmark_header) for landmark_header in landmarks_headers]

    landmarks_3d_headers = ["X_" + str(i) for i in range(n_landmarks)]
    landmarks_3d_headers.extend(["Y_" + str(i) for i in range(n_landmarks)])
    landmarks_3d_headers.extend(["Z_" + str(i) for i in range(n_landmarks)])
    landmark_3d_column_idxs = [headers.index(landmark_header) for landmark_header in landmarks_3d_headers]

    return landmark_column_idxs, landmark_3d_column_idxs

def get_pose_column_idxs_from_csv(csv_file_path):
    pose_column_headers = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]

    with open(csv_file_path, "r") as fp:
        headers = fp.readline().split(",")

    #Remove leading and trailing spaces from header names
    headers = [header.strip() for header in headers]
    pose_column_idxs = [headers.index(pose_header) for pose_header in pose_column_headers]

    return pose_column_idxs

def get_similarity_transform(destination_landmarks, source_landmarks):
    dest_mean = np.mean(destination_landmarks, axis=0)
    src_mean = np.mean(source_landmarks, axis=0)

    src_vec = (source_landmarks - src_mean).flatten()
    dest_vec = (destination_landmarks - dest_mean).flatten()

    a = np.dot(src_vec, dest_vec) / np.linalg.norm(src_vec)**2
    b = 0
    for i in range(destination_landmarks.shape[0]):
        b += src_vec[2*i] * dest_vec[2*i+1] - src_vec[2*i+1] * dest_vec[2*i]
    b = b / np.linalg.norm(src_vec)**2

    T = np.array([[a, -b], [b, a]])
    src_mean = np.dot(T, src_mean)

    return T, dest_mean - src_mean

def align_image(img, landmarks, output_shape, canonical_landmarks):
    A, t = get_similarity_transform(landmarks, canonical_landmarks)

    M = np.hstack((A, t[:, np.newaxis]))
    M = cv2.invertAffineTransform(M)

    out_img = cv2.warpAffine(img, M, output_shape[:2])

    return out_img

def parse_celeba_attribute_file(file_path):
    with open(file_path, "r") as fp:
        lines = fp.readlines()

    attribute_names = lines[1].split()
    attribute_labels = {}
    for line in lines[2:]:
        split_line = line.split()
        image_filename = split_line[0]
        image_filename = os.path.splitext(image_filename)[0]
        image_attributes = [0 if x == "-1" else 1 for x in split_line[1:]]

        attribute_labels[image_filename] = dict(zip(attribute_names, image_attributes))

    return attribute_labels

def run_openface_on_dir(input_dir, openface_path):
    done_file_path = os.path.join(input_dir, "landmarks_detected")
    if not os.path.exists(done_file_path): # Check if "landmarks_detected" file exists
        openface_output_dir = os.path.join(input_dir, "processed")
        os.makedirs(openface_output_dir, exist_ok=True)
        if os.path.exists(openface_path):
            print("Running OpenFace on data dir %s"%(input_dir))
            subprocess.call([openface_path, "-fdir", input_dir, "-wild", "-out_dir", openface_output_dir, "-2Dfp", "-3Dfp", "-pose", "-multi_view 1"])
        else:
            raise ImportError("OpenFace not found, please download it using the download_deps.py script.")

        # create file indicating that the directory has been processed
        with open(done_file_path, "w+") as _:
            pass

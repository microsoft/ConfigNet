# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Utility scripts for the ConfigNet repo"""
import numpy as np
import tensorflow as tf
import glob
import os
import sys
from matplotlib import pyplot as plt
import json

from . import azure_ml_utils

def load_confignet(model_path):
    with open(model_path, "r") as fp:
        metadata = json.load(fp)
    model_type = metadata["model_type"]

    confignet_class = getattr(sys.modules["confignet"], model_type)

    return confignet_class.load(model_path)

def draw_loss_grid(losses, loss_names, pix_per_plot=300):
    assert len(losses) == len(loss_names)
    n_losses = len(loss_names)
    square_size = int(np.ceil(np.sqrt(n_losses)))

    dpi = 100
    pix_size = square_size * pix_per_plot
    plt.figure(figsize=(pix_size // dpi, pix_size // dpi), dpi=dpi)

    for i in range(n_losses):
        plt.subplot(square_size, square_size, i + 1)
        plt.title(loss_names[i])
        plt.semilogy(losses[i])

    plt.tight_layout()

def merge_configs(default_config, input_config):
    """Recursively merge configuration dictionaries"""
    result = {}
    for name in default_config:
        lhs_value = default_config[name]
        if name in input_config:
            rhs_value = input_config[name]
            if isinstance(lhs_value, dict):
                assert isinstance(rhs_value, dict)
                result[name] = merge_configs(lhs_value, rhs_value)
            else:
                result[name] = rhs_value
        else:
            result[name] = lhs_value

    for name in input_config:
        rhs_value = input_config[name]
        if isinstance(rhs_value, dict) and name in default_config.keys():
            continue

        result[name] = rhs_value

    return result

def transform_3d_grid_tf(grid, transform):
    assert(grid.shape[1] == grid.shape[2] == grid.shape[3])
    grid_size = int(grid.shape[1])
    grid_center = (grid_size - 1) / 2
    batch_size = tf.shape(grid)[0]
    num_grid_points = grid_size ** 3

    xs, ys, zs = np.meshgrid(range(grid_size), range(grid_size), range(grid_size), indexing="ij")
    grid_coords = np.vstack((xs.flatten(), ys.flatten(), zs.flatten()))

    grid_coords = tf.cast(grid_coords, tf.float32)
    grid_coords = tf.tile(tf.expand_dims(grid_coords, 0), (batch_size, 1, 1))
    transform = tf.cast(transform, tf.float32)
    grid = tf.cast(grid, tf.float32)

    transformed_coords = tf.matmul(transform, grid_coords - grid_center) + grid_center
    transformed_coords = tf.transpose(transformed_coords, (1, 0, 2))
    transformed_coords = tf.clip_by_value(transformed_coords, 0, grid_size - 1)
    transformed_coords = tf.reshape(transformed_coords, (int(transformed_coords.shape[0]), -1))

    transformed_coords_floor = tf.floor(transformed_coords)
    transformed_coords_floor = tf.clip_by_value(transformed_coords_floor, 0, grid_size - 1)
    transformed_coords_ceil = transformed_coords_floor + 1
    transformed_coords_ceil = tf.clip_by_value(transformed_coords_ceil, 0, grid_size - 1)

    transformed_coords_floor_int = tf.cast(transformed_coords_floor, tf.int32)
    transformed_coords_ceil_int = tf.cast(transformed_coords_ceil, tf.int32)


    batch_axis_idxs = tf.tile(tf.expand_dims(tf.range(batch_size), 1), (1, num_grid_points))
    batch_axis_idxs = tf.reshape(batch_axis_idxs, [-1])

    c000 = tf.gather_nd(grid, tf.stack((batch_axis_idxs, transformed_coords_floor_int[0], transformed_coords_floor_int[1], transformed_coords_floor_int[2]), axis=1))
    c100 = tf.gather_nd(grid, tf.stack((batch_axis_idxs, transformed_coords_ceil_int[0], transformed_coords_floor_int[1], transformed_coords_floor_int[2]), axis=1))
    c101 = tf.gather_nd(grid, tf.stack((batch_axis_idxs, transformed_coords_ceil_int[0], transformed_coords_floor_int[1], transformed_coords_ceil_int[2]), axis=1))
    c001 = tf.gather_nd(grid, tf.stack((batch_axis_idxs, transformed_coords_floor_int[0], transformed_coords_floor_int[1], transformed_coords_ceil_int[2]), axis=1))

    c010 = tf.gather_nd(grid, tf.stack((batch_axis_idxs, transformed_coords_floor_int[0], transformed_coords_ceil_int[1], transformed_coords_floor_int[2]), axis=1))
    c110 = tf.gather_nd(grid, tf.stack((batch_axis_idxs, transformed_coords_ceil_int[0], transformed_coords_ceil_int[1], transformed_coords_floor_int[2]), axis=1))
    c111 = tf.gather_nd(grid, tf.stack((batch_axis_idxs, transformed_coords_ceil_int[0], transformed_coords_ceil_int[1], transformed_coords_ceil_int[2]), axis=1))
    c011 = tf.gather_nd(grid, tf.stack((batch_axis_idxs, transformed_coords_floor_int[0], transformed_coords_ceil_int[1], transformed_coords_ceil_int[2]), axis=1))

    diffs = transformed_coords - transformed_coords_floor
    diffs = tf.expand_dims(diffs, -1)

    c00 = c000 * (1 - diffs[0]) + c100 * diffs[0]
    c01 = c001 * (1 - diffs[0]) + c101 * diffs[0]
    c10 = c010 * (1 - diffs[0]) + c110 * diffs[0]
    c11 = c011 * (1 - diffs[0]) + c111 * diffs[0]

    c0 = c00 * (1 - diffs[1]) + c10 * diffs[1]
    c1 = c01 * (1 - diffs[1]) + c11 * diffs[1]

    weighted_sum = c0 * (1 - diffs[2]) + c1 * diffs[2]

    output_grid = tf.reshape(weighted_sum, (-1, *grid.shape[1:]))

    return output_grid

def euler_angles_to_matrix(angle_vector):
    angles = tf.reshape(angle_vector,[-1,3])
    N = angles.shape[0]

    sins = tf.sin(angles)
    coss = tf.cos(angles)

    a11 = coss[:,2] * coss[:,1]
    a12 = -sins[:,2]
    a13 = coss[:,2] * sins[:,1]
    a21 = sins[:,0] * sins[:,1] + coss[:,0] * coss[:,1] * sins[:,2]
    a22 = coss[:,0] * coss[:,2]
    a23 = coss[:,0] * sins[:,2] * sins[:,1] - coss[:,1]*sins[:,0]
    a31 = coss[:,1] * sins[:,0] * sins[:,2] - coss[:,0]*sins[:,1]
    a32 = coss[:,2] * sins[:,0]
    a33 = coss[:,0]*coss[:,1] + sins[:,0]*sins[:,1] *sins[:,2]

    mat9 = tf.stack([a11, a12, a13, a21, a22, a23, a31, a32, a33], axis=-1)
    mat33 = tf.reshape(mat9, [-1, 3, 3])

    if N==1:
        tf.squeeze(mat33, axis=0)

    return mat33

def get_layer_style(image_features, eps=1e-6):
    normalization_axes = None
    if len(image_features.shape) == 4:
        normalization_axes = [1, 2]
    elif len(image_features.shape) == 5:
        normalization_axes = [1, 2, 3]
    else:
        raise NotImplementedError()

    mean = tf.reduce_mean(image_features, axis=normalization_axes, keepdims=True)
    std = tf.sqrt(tf.reduce_mean(tf.square(image_features - mean), axis=normalization_axes, keepdims=True) + eps)

    return mean, std

def attempt_reloading_checkpoint(output_dir, dnn_loader):
    potential_checkpoint_dirs = [os.path.join(output_dir, "checkpoints")]
    if "PT_PREV_OUTPUT_DIR" in os.environ.keys():
        potential_checkpoint_dirs.append(os.path.join(os.environ["PT_PREV_OUTPUT_DIR"], "checkpoints"))

    print("Attempting to restart job from checkpoint. Potential checkpoint dirs are:")
    for potential_checkpoint_dir in potential_checkpoint_dirs:
        print(potential_checkpoint_dir)

    for checkpoint_dir in potential_checkpoint_dirs:
        if not os.path.exists(checkpoint_dir):
            continue

        checkpoint_filenames = glob.glob(os.path.join(checkpoint_dir, "*.json"))
        if len(checkpoint_filenames) == 0:
            continue
        print("Found loadable checkpoint")

        dnn = dnn_loader(checkpoint_filenames[-1])
        return dnn

def build_image_matrix(images, n_rows, n_cols):
    image_shape = images.shape[1:]

    image_matrix = np.zeros((n_rows * image_shape[0], n_cols * image_shape[1], 3), dtype=np.uint8)
    for i in range(n_cols):
        for j in range(n_rows):
            image_matrix[j * image_shape[0] : (j + 1) * image_shape[0], i * image_shape[1] : (i + 1) * image_shape[1]] = images[j * n_cols + i]

    return image_matrix

def wrap_with_list(var):
    if isinstance(var, list):
        return var
    else:
        return [var]

def flip_random_subset_of_images(images):
    flip_or_not = np.random.randint(0, 2, size=images.shape[0])
    for i, flip in enumerate(flip_or_not):
        if flip == 1:
            images[i] = np.fliplr(images[i])

    return images

def update_loss_dict(main_loss_dict, new_loss_dict):
    for key, val in new_loss_dict.items():
        val = float(val)
        if key in main_loss_dict.keys():
            main_loss_dict[key].append(val)
        else:
            main_loss_dict[key] = [val]

def log_loss_vals(loss_dict, output_dir, step_number, prefix, aml_run=None, tb_log_writer=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    loss_names = list(loss_dict.keys())
    loss_vals = list(loss_dict.values())
    most_recent_loss_vals = [x[-1] for x in loss_vals]

    if aml_run is not None:
        azure_ml_utils.log_losses(aml_run, loss_names, most_recent_loss_vals, prefix)
    else:
        draw_loss_grid(loss_vals, loss_names)
        plt.savefig(os.path.join(output_dir, prefix + "losses.png"))
        plt.close()
        plt.semilogy(loss_dict["loss_sum"])
        plt.savefig(os.path.join(output_dir, prefix + "loss_sum.png"))
        plt.close()

    if tb_log_writer is not None:
        # Replace last _ with /
        prefix_for_tb = prefix[::-1].replace("_", "/", 1)[::-1]
        with tb_log_writer.as_default():
            for name, value in zip(loss_names, most_recent_loss_vals):
                tf.summary.scalar(prefix_for_tb + name, float(value), step=step_number)

    all_loss_vals = np.stack(loss_vals, axis=1)
    header = "\t".join(loss_names)
    np.savetxt(os.path.join(output_dir, prefix + "losses.txt"), all_loss_vals, header=header)

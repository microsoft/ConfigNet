# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
""" Scripts for computing GAN metrics (KID and FID) """
from tensorflow.keras.applications import inception_v3
from sklearn.metrics.pairwise import polynomial_kernel
import numpy as np
import scipy

class InceptionFeatureExtractor:
    def __init__(self, input_shape):
        self.model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling="avg")

    def get_features(self, images, max_chunk_size=1000):
        n_imgs = images.shape[0]
        features = np.zeros((n_imgs, *self.model.output_shape[1:]), np.float32)

        n_chunks = 1 + n_imgs // max_chunk_size
        for i in range(n_chunks):
            chunk_begin = i * max_chunk_size
            chunk_end = min((i + 1) * max_chunk_size, chunk_begin + n_imgs - chunk_begin)

            if chunk_end - chunk_begin == 0:
                break
            pre_processed_images = inception_v3.preprocess_input(images[chunk_begin : chunk_end])
            features[chunk_begin : chunk_end] = self.model.predict(pre_processed_images)

        return features

def compute_FID(features_g, features_r):
    mean_g = np.mean(features_g, axis=0)
    mean_r = np.mean(features_r, axis=0)

    cov_g = np.cov(features_g, rowvar=False)
    cov_r = np.cov(features_r, rowvar=False)

    centroid_distance = np.linalg.norm(mean_g - mean_r)**2
    covariance_distance = np.trace(cov_g + cov_r - 2 * scipy.linalg.sqrtm(np.dot(cov_g, cov_r)))

    covariance_distance = np.real(covariance_distance)

    score = centroid_distance + covariance_distance

    return score

def compute_KID(features_g, features_r):
    kernel_gen_gen = polynomial_kernel(features_g, degree=3, coef0=1.0)
    kernel_real_real = polynomial_kernel(features_r, degree=3, coef0=1.0)
    kernel_gen_real = polynomial_kernel(features_g, features_r, degree=3, coef0=1.0)

    # Eq. 4 in https://arxiv.org/pdf/1801.01401.pdf
    m = features_g.shape[0]
    n = features_r.shape[0]
    term1 = (1 / (m * (m - 1))) * (np.sum(kernel_gen_gen) - np.sum(np.diagonal(kernel_gen_gen)))
    term2 = (1 / (n * (n - 1))) * (np.sum(kernel_real_real) - np.sum(np.diagonal(kernel_real_real)))
    term3 = (1 / (m * n)) * np.sum(kernel_gen_real)

    kid = term1 + term2 - 2 * term3

    return kid

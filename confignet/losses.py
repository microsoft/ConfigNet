# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tensorflow as tf
from tensorflow import keras
import numpy as np

def GAN_G_loss(scores):
    return keras.backend.mean(tf.math.softplus(-scores))

def GAN_D_loss(labels, scores):
    return keras.backend.mean(labels * tf.math.softplus(-scores) + (1.0 - labels) * tf.math.softplus(scores))

def eye_loss(gt_imgs, gen_imgs, eye_masks):
    img_diff = (gt_imgs - gen_imgs) * tf.expand_dims(tf.convert_to_tensor(eye_masks, tf.float32), -1)

    loss_val_per_img = tf.reduce_sum(tf.square(img_diff), axis=(1, 2, 3)) / (1 + np.sum(eye_masks, axis=(1, 2)))

    return tf.reduce_mean(loss_val_per_img)

def compute_discriminator_loss(discriminator, real_imgs, fake_imgs):
    # Labels
    valid_y = tf.ones((real_imgs.shape[0], 1), dtype=tf.float32)
    fake_y = tf.zeros((fake_imgs.shape[0], 1), dtype=tf.float32)

    losses = {}
    with tf.GradientTape(persistent=True) as grad_reg_tape:
        grad_reg_tape.watch(real_imgs)
        discriminator_output_real = discriminator(real_imgs)
    discriminator_output_fake = discriminator(fake_imgs)

    # GAN loss on real
    for i, (disc_out) in enumerate(discriminator_output_real.values()):
        gan_loss = GAN_D_loss(valid_y, disc_out)
        losses["GAN_loss_real_" + str(i)] = gan_loss

    # GAN loss on fake
    for i, disc_out in enumerate(discriminator_output_fake.values()):
        gan_loss = GAN_D_loss(fake_y, disc_out)
        losses["GAN_loss_fake_" + str(i)] = gan_loss

    # Gradient penalty
    for i, single_discr_output in enumerate(discriminator_output_real.values()):
        losses["gp_loss_" + str(i)] = gradient_regularization(grad_reg_tape, single_discr_output, real_imgs)

    losses["loss_sum"] = tf.reduce_sum(list(losses.values()))

    return losses

def compute_latent_discriminator_loss(latent_discriminator, real_latents, fake_latents):
    batch_size = real_latents.shape[0]

    # Labels
    valid_y = np.ones((batch_size, 1))
    fake_y = np.zeros((batch_size, 1))

    losses = {}
    with tf.GradientTape(persistent=True) as grad_reg_tape:
        grad_reg_tape.watch(real_latents)
        discriminator_output_real = latent_discriminator(real_latents)
    discriminator_output_fake = latent_discriminator(fake_latents)

    # GAN loss on real
    gan_loss = GAN_D_loss(valid_y, discriminator_output_real)
    losses["GAN_loss_real"] = gan_loss
    # GAN loss on fake
    gan_loss = GAN_D_loss(fake_y, discriminator_output_fake)
    losses["GAN_loss_fake"] = gan_loss
    # Gradient penalty
    losses["gp_loss"] = gradient_regularization(grad_reg_tape, discriminator_output_real, real_latents)

    losses["loss_sum"] = tf.reduce_sum(list(losses.values()))

    return losses

def gradient_regularization(grad_reg_tape, real_out, real_in):
    gradients_wrt_input = grad_reg_tape.gradient(real_out, real_in)
    gradients_sqr = tf.square(gradients_wrt_input)
    r1_penalty = tf.reduce_sum(gradients_sqr, axis=range(1, len(gradients_sqr.shape)))
    r1_penalty = tf.reduce_mean(r1_penalty)

    # weights from L. Mescheder et al.: Which Training Methods for GANs do actually Converge
    return 10 * 0.5 * r1_penalty

# Known as identity loss in HoloGAN
def compute_latent_regression_loss(generator_outputs, labels, latent_regressor):
    latent_regressor_output = latent_regressor(generator_outputs)
    latent_regression_loss = tf.losses.mean_squared_error(labels, latent_regressor_output)
    latent_regression_loss = tf.reduce_mean(latent_regression_loss)

    return latent_regression_loss
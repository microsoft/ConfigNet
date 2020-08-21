# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import sys
import argparse
import json
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import cv2

DEFAULT_CONFIG = {
    "input_shape": None,
    "predicted_attributes": None,
    "optimizer": {
        "lr": 0.001
    },
    "batch_size": 32
}

class CelebaAttributeClassifier:
    def __init__(self, config):
        self.config = config

        self.logs = {}

        self.classifier = None

        self.initialize_dnn()

    def save(self, output_dir, output_filename):
        weights = self.classifier.get_weights()
        metadata = {}
        metadata["logs"] = self.logs
        metadata["config"] = self.config

        with open(os.path.join(output_dir, output_filename + ".json"), "w") as fp:
            json.dump(metadata, fp, indent=4)
        np.save(os.path.join(output_dir, output_filename + ".npy"), weights)

    @classmethod
    def load(cls, file_path):
        with open(file_path, "r") as fp:
            metadata = json.load(fp)

        weight_file_path = os.path.splitext(file_path)[0] + ".npy"
        weights = np.load(weight_file_path, allow_pickle=True)

        classifier = cls(metadata["config"])
        classifier.logs = metadata["logs"]
        classifier.classifier.set_weights(weights)

        return classifier

    def initialize_dnn(self):
        base_model = keras.applications.MobileNetV2(input_shape=self.config["input_shape"], include_top=False)

        self.classifier = keras.Sequential()
        self.classifier.add(base_model)
        self.classifier.add(keras.layers.GlobalAveragePooling2D())
        self.classifier.add(keras.layers.BatchNormalization())
        self.classifier.add(keras.layers.Dropout(0.5))
        self.classifier.add(keras.layers.Dense(len(self.config["predicted_attributes"]), activation="sigmoid"))

    def callback(self, epoch, logs, output_dir):
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.save(checkpoint_dir, str(epoch).zfill(4))

        for key, value in logs.items():
            if key not in self.logs.keys():
                self.logs[key] = []
            self.logs[key].append(float(value))

        if len(self.logs["val_binary_accuracy"]) == 1 or self.logs["val_binary_accuracy"][-1] > np.max(self.logs["val_binary_accuracy"][:-1]):
            best_model_dir = os.path.join(output_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            self.save(best_model_dir, str(epoch).zfill(4))


        plt.plot(self.logs["loss"])
        plt.plot(self.logs["val_loss"])
        plt.savefig(os.path.join(output_dir, "losses.png"))
        plt.clf()

        plt.plot(self.logs["binary_accuracy"])
        plt.plot(self.logs["val_binary_accuracy"])
        plt.savefig(os.path.join(output_dir, "metrics.png"))
        plt.clf()

        log_names = list(self.logs.keys())
        log_vals = list(self.logs.values())

        all_log_vals = np.stack(log_vals, axis=1)
        header = "\t".join(log_names)
        np.savetxt(os.path.join(output_dir, "logs.txt"), all_log_vals, header=header)

    def sample_batch_from_dataset(self, dataset, batch_size=None, add_noise=False):
        if batch_size is None:
            batch_size = self.config["batch_size"]
        sample_idxs = np.random.randint(0, dataset.imgs.shape[0], batch_size)
        imgs = np.copy(dataset.imgs[sample_idxs])
        imgs = keras.applications.mobilenet_v2.preprocess_input(imgs)
        if add_noise:
            imgs[:batch_size // 2] = imgs[:batch_size // 2] + np.random.normal(0, 0.05, imgs[:batch_size // 2].shape)
        attributes = dataset.get_attribute_values(sample_idxs, self.config["predicted_attributes"])

        return imgs, attributes

    def batch_generator(self, training_set):
        while True:
            yield self.sample_batch_from_dataset(training_set, add_noise=False)

    def train(self, training_set, validation_set, output_dir, n_epochs, steps_per_epoch):
        optimizer = keras.optimizers.Adam(**self.config["optimizer"])
        self.classifier.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])

        callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda epochs, logs: self.callback(epochs, logs, output_dir))
        validation_imgs, validation_labels = self.sample_batch_from_dataset(validation_set, 200)

        self.classifier.fit_generator(
            self.batch_generator(training_set),
            steps_per_epoch=steps_per_epoch,
            epochs=n_epochs,
            callbacks=[callback],
            validation_data=(validation_imgs, validation_labels),
        )

    def predict_attributes(self, input_images):
        if input_images.dtype == np.float32:
            input_images = (input_images + 1) * 127.5
        if input_images.shape[1:] != self.config["input_shape"]:
            resized_input_images = np.zeros((input_images.shape[0], *self.config["input_shape"]), dtype=input_images.dtype)
            desired_img_size_x_y = tuple(self.config["input_shape"][:2][::-1])
            for i, img in enumerate(input_images):
                resized_input_images[i] = cv2.resize(img, desired_img_size_x_y)
            input_images = resized_input_images

        preprocessed_images = keras.applications.mobilenet_v2.preprocess_input(input_images)
        attribute_probabilities = self.classifier.predict(preprocessed_images)

        return attribute_probabilities
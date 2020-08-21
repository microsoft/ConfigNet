# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import json
import cv2

from .inception_distance import InceptionFeatureExtractor, compute_FID, compute_KID
from .celeba_attribute_prediction import CelebaAttributeClassifier
from .controllability_metric_configs import ControllabilityMetricConfigs
from .blendshape_names import blendshape_names

class ControllabilityMetrics:
    def __init__(self, confinet_model, attribute_classifier, per_image_tuning_iters=0):
        self.confinet_model = confinet_model

        if isinstance(attribute_classifier, CelebaAttributeClassifier):
            self.attribute_classifier = attribute_classifier
        else:
            # Assume the CelebaAttributeClassifier can be loaded from the specified input
            self.attribute_classifier = CelebaAttributeClassifier.load(attribute_classifier)

        self.per_image_tuning_iters = per_image_tuning_iters
        if confinet_model is not None:
            self.facemodel_param_names = list(self.confinet_model.config["facemodel_inputs"].keys())

    def get_facemodel_params_for_config(self, attribute_config, other_param):
        facemodel_params = self.confinet_model.sample_facemodel_params(1)

        if other_param:
            param_value = attribute_config.facemodel_param_value_other
        else:
            param_value = attribute_config.facemodel_param_value

        param_idx = self.facemodel_param_names.index(attribute_config.facemodel_param_name)
        if isinstance(param_value, dict):
            if attribute_config.facemodel_param_name == "blendshape_values":
                # labels for each dimension of given facemodel param
                facemodel_params[param_idx][:] = 0
                for key, value in param_value.items():
                    idx = blendshape_names.index(key)
                    facemodel_params[param_idx][:, idx] = value
            else:
                raise NotImplementedError
        else:
            facemodel_params[param_idx][:] = param_value

        return facemodel_params

    def get_images_for_controllable_attribute(self, attribute_config, latent_vectors, rotations, other_param=False):
        facemodel_params = self.get_facemodel_params_for_config(attribute_config, other_param)
        latent_vector_with_attribute_set = self.confinet_model.synthetic_encoder(facemodel_params)

        modified_param_idx = self.facemodel_param_names.index(attribute_config.facemodel_param_name)
        facemodel_param_dims = list(self.confinet_model.config["facemodel_inputs"].values())
        start_idx = int(np.sum([x[1] for x in facemodel_param_dims[:modified_param_idx]]))
        end_idx = start_idx + facemodel_param_dims[modified_param_idx][1]

        modified_latent_vectors = np.copy(latent_vectors)
        modified_latent_vectors[:, start_idx : end_idx] = latent_vector_with_attribute_set[0, start_idx : end_idx]

        output_imgs = self.confinet_model.generate_images(modified_latent_vectors, rotations)

        return output_imgs

    def generate_images_for_metric(self, input_images):
        if self.per_image_tuning_iters > 0:
            raw_decoded_images = []
            images_with_attributes = {}
            images_without_attributes = {}
            for config_name, _ in ControllabilityMetricConfigs.all_configs():
                images_with_attributes[config_name] = []
                images_without_attributes[config_name] = []

            for img in input_images:
                img = img[np.newaxis]
                latent_vectors, rotations = self.confinet_model.fine_tune_on_img(img, n_iters=self.per_image_tuning_iters)
                raw_decoded_images.append(self.confinet_model.generate_images(latent_vectors, rotations)[0])

                for config_name, attribute_config in ControllabilityMetricConfigs.all_configs():
                    image_with_attribute = self.get_images_for_controllable_attribute(attribute_config, latent_vectors, rotations)[0]
                    image_without_attribute = self.get_images_for_controllable_attribute(attribute_config, latent_vectors, rotations, other_param=True)[0]
                    images_with_attributes[config_name].append(image_with_attribute)
                    images_without_attributes[config_name].append(image_without_attribute)

            raw_decoded_images = np.array(raw_decoded_images)
            for key in images_with_attributes.keys():
                images_with_attributes[key] = np.array(images_with_attributes[key])
            for key in images_without_attributes.keys():
                images_without_attributes[key] = np.array(images_without_attributes[key])
        else:
            latent_vectors, rotations = self.confinet_model.encode_images(input_images)
            raw_decoded_images = self.confinet_model.generate_images(latent_vectors, rotations)
            images_with_attributes = {}
            images_without_attributes = {}
            for config_name, attribute_config in ControllabilityMetricConfigs.all_configs():
                images_with_attribute = self.get_images_for_controllable_attribute(attribute_config, latent_vectors, rotations)
                images_without_attribute = self.get_images_for_controllable_attribute(attribute_config, latent_vectors, rotations, other_param=True)
                images_with_attributes[config_name] = images_with_attribute
                images_without_attributes[config_name] = images_without_attribute

        return raw_decoded_images, images_with_attributes, images_without_attributes

    def get_metrics_for_attribute_pairs(self, set_attributes, not_set_attributes, attribute_config):
        attribute_names = self.attribute_classifier.config["predicted_attributes"]

        # This attribute is expected to change
        driven_attribute_idx = attribute_names.index(attribute_config.driven_attribute)
        # These attributes are not expected to be constant
        changing_attribute_names = attribute_config.ignored_attributes + [attribute_config.driven_attribute]
        # These attributes are expected to be constant
        constant_attribute_idxs = [i for i, attribute_name in enumerate(attribute_names) if attribute_name not in changing_attribute_names]

        mean_value_of_attribute_after_setting = np.mean(set_attributes[:, driven_attribute_idx])
        mean_value_of_attribute_after_setting_other = np.mean(not_set_attributes[:, driven_attribute_idx])
        #mean_value_of_attribute_before_change = np.mean(original_attributes[:, driven_attribute_idx])

        n_samples = len(set_attributes)
        assert n_samples == len(not_set_attributes)
        attribute_values_for_correlation = np.hstack((np.ones(n_samples), np.zeros(n_samples)))
        predicted_values_for_correlation = np.hstack((set_attributes[:, driven_attribute_idx], not_set_attributes[:, driven_attribute_idx]))
        corr_coef = np.corrcoef(np.vstack((attribute_values_for_correlation, predicted_values_for_correlation)))


        mad_of_const_attributes = np.mean(np.abs(set_attributes[:, constant_attribute_idxs] - not_set_attributes[:, constant_attribute_idxs]), axis=0)
        mad_of_const_attributes = np.mean(mad_of_const_attributes)

        return float(mean_value_of_attribute_after_setting), float(mean_value_of_attribute_after_setting_other), float(mad_of_const_attributes), float(corr_coef[0, 1])

    def get_metrics_for_attribute_config(self, attribute_config, images_with_attribute, images_without_attribute):
        set_attributes = self.attribute_classifier.predict_attributes(images_with_attribute)
        not_set_attributes = self.attribute_classifier.predict_attributes(images_without_attribute)
        metrics = self.get_metrics_for_attribute_pairs(set_attributes, not_set_attributes, attribute_config)

        return metrics

    def get_metrics(self, input_images, img_output_dir=None):
        raw_decoded_images, images_with_attributes, images_without_attributes = self.generate_images_for_metric(input_images)
        if img_output_dir is not None:
            os.makedirs(img_output_dir, exist_ok=True)
            for i in range(len(input_images)):
                cv2.imwrite(os.path.join(img_output_dir, "gt_img_%04d.png"%i), input_images[i])
                cv2.imwrite(os.path.join(img_output_dir, "raw_img_%04d.png"%i), raw_decoded_images[i])
                for config_name, _ in ControllabilityMetricConfigs.all_configs():
                    attr_img = images_with_attributes[config_name][i]
                    img_name = "%s_img_%04d.png"%(config_name, i)
                    cv2.imwrite(os.path.join(img_output_dir, img_name), attr_img)

                    no_attr_img = images_without_attributes[config_name][i]
                    img_name = "%s_img_not_set_%04d.png"%(config_name, i)
                    cv2.imwrite(os.path.join(img_output_dir, img_name), no_attr_img)

        return self.get_metrics_from_attribute_images(images_with_attributes, images_without_attributes)

    def get_metrics_from_attribute_images(self, images_with_attributes, images_without_attributes):
        metrics = {}
        for config_name, attribute_config in ControllabilityMetricConfigs.all_configs():
            attribute_metrics = self.get_metrics_for_attribute_config(attribute_config, images_with_attributes[config_name],
                                                                      images_without_attributes[config_name])
            metrics[config_name] = attribute_metrics

        metrics["contr_attribute_means"] = tuple(np.mean(list(metrics.values()), axis=0))
        # Weights chosen basen on perceived importance
        metrics["controllability"] = 10 * metrics["contr_attribute_means"][2] + (1 - metrics["contr_attribute_means"][0])

        return metrics

    def update_and_log_metrics(self, images, metrics_dict, output_dir, aml_run=None, tb_log_writer=None):
        os.makedirs(output_dir, exist_ok=True)

        new_metrics = self.get_metrics(images)

        for key, value in new_metrics.items():
            if key not in metrics_dict.keys():
                metrics_dict[key] = []
            metrics_dict[key].append(value)

        if aml_run is not None:
            for key, value in new_metrics.items():
                aml_run.log(key, value)
        if tb_log_writer is not None:
            with tb_log_writer.as_default():
                for key, value in new_metrics.items():
                    if isinstance(value, tuple):
                        if key == "contr_attribute_means":
                            prefix = "metrics/"
                        else:
                            prefix = "contr_metrics_per_attribute/"
                        tf.summary.scalar(prefix + key + "_post", value[0], step=metrics_dict["training_step_number"][-1])
                        tf.summary.scalar(prefix + key + "_pre", value[1], step=metrics_dict["training_step_number"][-1])
                        tf.summary.scalar(prefix + key + "_other", value[2], step=metrics_dict["training_step_number"][-1])
                    else:
                        tf.summary.scalar("metrics/" + key, value, step=metrics_dict["training_step_number"][-1])

        metrics_dict_contr_only = dict([(key, metrics_dict[key]) for key in new_metrics.keys()])
        with open(os.path.join(output_dir, "controllability_metrics.json"), "w") as fp:
            json.dump(metrics_dict_contr_only, fp, indent=4)

class InceptionMetrics:
    def __init__(self, confignet_config, dataset, n_samples_for_metrics=1000):
        self.n_samples_for_metrics = n_samples_for_metrics
        self.inception_feature_extractor = InceptionFeatureExtractor(confignet_config["output_shape"])

        metric_sample_idxs = np.random.randint(0, dataset.imgs.shape[0], n_samples_for_metrics)
        self.gt_inception_features = dataset.inception_features[metric_sample_idxs]

    def get_metrics(self, generated_images):
        generated_inception_features = self.inception_feature_extractor.get_features(generated_images)
        kid = compute_KID(generated_inception_features, self.gt_inception_features)
        fid = compute_FID(generated_inception_features, self.gt_inception_features)

        return kid, fid

    def update_and_log_metrics(self, images, metrics_dict, output_dir, aml_run=None, tb_log_writer=None):
        os.makedirs(output_dir, exist_ok=True)

        kid, fid = self.get_metrics(images)

        if "kid" not in metrics_dict.keys():
            metrics_dict["kid"] = []
        if "fid" not in metrics_dict.keys():
            metrics_dict["fid"] = []

        metrics_dict["kid"].append(kid)
        metrics_dict["fid"].append(fid)

        assert len(metrics_dict["kid"]) == len(metrics_dict["fid"])

        if "training_step_number" in metrics_dict:
            step_numbers_for_logs = metrics_dict["training_step_number"]
            assert len(step_numbers_for_logs) == len(metrics_dict["kid"])
        else:
            step_numbers_for_logs = range(len(metrics_dict["kid"]))


        if aml_run is not None:
            aml_run.log("Kernel Inception Distance", kid)
            aml_run.log("Frechet Inception Distance", fid)
        else:
            ax = plt.gca()
            color = "tab:blue"
            ax.set_ylabel("KID", color=color)
            ax.semilogy(step_numbers_for_logs, metrics_dict["kid"], color=color)
            ax.tick_params(axis='y', labelcolor=color)

            ax = ax.twinx()
            color = "tab:red"
            ax.set_ylabel("FID", color=color)
            ax.semilogy(step_numbers_for_logs, metrics_dict["fid"], color=color)
            ax.tick_params(axis='y', labelcolor=color)

            plt.savefig(os.path.join(output_dir, "inception_metrics.png"))
            plt.clf()

        if tb_log_writer is not None:
            with tb_log_writer.as_default():
                tf.summary.scalar("metrics/kid", kid, step=step_numbers_for_logs[-1])
                tf.summary.scalar("metrics/fid", fid, step=step_numbers_for_logs[-1])

        header = "\t".join(["step_number", "kid", "fid"])
        metrics_values_for_txt = np.stack((step_numbers_for_logs, metrics_dict["kid"], metrics_dict["fid"]), axis=1)

        np.savetxt(os.path.join(output_dir, "inception_metrics.txt"), metrics_values_for_txt, header=header)
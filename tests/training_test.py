""" Tests of Confignet training scripts """
import os
import sys

from fixtures import temporary_output_dir, test_asset_dir, model_dir

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import train_confignet
import train_latent_gan
import train_attribute_classifier

def test_confignet_training(temporary_output_dir, test_asset_dir, model_dir):
    training_set_path = os.path.join(test_asset_dir, "test_dataset_res_256.pck")
    model_output_dir = os.path.join(temporary_output_dir, "confignet")
    attribute_classifier_path = os.path.join(model_dir, "attribute_classifier", "model.json")


    # Run the training script with the pre-generated dataset
    script_arguments = "--real_training_set_path %s --synth_training_set_path %s --output_dir %s"%(training_set_path, training_set_path, model_output_dir)
    script_arguments += " --validation_set_path %s --attribute_classifier_path %s"%(training_set_path, attribute_classifier_path)
    script_arguments += " --stage_1_training_steps 1 --stage_2_training_steps 1 --batch_size 4 --n_samples_for_metrics 10"
    train_confignet.parse_args(script_arguments.split(" "))

def test_latent_gan_training(temporary_output_dir, test_asset_dir, model_dir):
    model_path = os.path.join(model_dir, "confignet_256", "model.json")
    training_set_path = os.path.join(test_asset_dir, "test_dataset_res_256.pck")

    script_arguments = "--confignet_path %s --training_set_path %s"%(model_path, training_set_path)
    script_arguments += " --output_dir %s --n_training_steps 1 --batch_size 4 --n_samples_for_metrics 10"%temporary_output_dir
    train_latent_gan.parse_args(script_arguments.split(" "))

def test_attribute_classifier_training(temporary_output_dir, test_asset_dir):
    training_set_path = os.path.join(test_asset_dir, "test_dataset_res_256.pck")

    script_arguments = "--training_set_path %s --validation_set_path %s --output_dir %s"%(training_set_path, training_set_path, temporary_output_dir)
    script_arguments += " --n_epochs 2 --steps_per_epoch 2 --batch_size 4"
    train_attribute_classifier.parse_args(script_arguments.split(" "))

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import numpy as np

class BasicUI:
    def __init__(self, confignet_model):
        self.confignet_model = confignet_model

        self.exit = False
        self.rotation_offset = np.zeros((1, 3))
        self.eye_rotation_offset = np.zeros((1, 3))
        self.controlled_param_idx = 0

        self.facemodel_param_names = list(self.confignet_model.config["facemodel_inputs"].keys())
        # remove eye rotation as in the demo it is controlled separately
        eye_rotation_param_idx = self.facemodel_param_names.index("bone_rotations:left_eye")
        self.facemodel_param_names.pop(eye_rotation_param_idx)

        self.render_input_interp_0 = None
        self.render_input_interp_1 = None

        self.rotation_angle_step_size = 0.05

        self.interpolation_coef = 1.0
        self.n_interpolation_steps = 5
        self.interpolation_step_length = 1.0 / self.n_interpolation_steps

        # HDRI turntable inputs
        hdri_turntable_file_path = os.path.join(os.path.dirname(__file__), "..", "assets","hdri_turntable_embeddings.npy")
        self.hdri_turntable_embeddings = np.load(hdri_turntable_file_path)
        self.current_hdri_embedding_frame = 0
        self.sweeping_hdri = False

    def perform_per_frame_actions(self):
        if self.interpolation_coef < 1.0:
            self.interpolation_coef += self.interpolation_step_length
            self.interpolation_coef = min(self.interpolation_coef, 1.0)

    def set_next_embeddings(self, embeddings):
        if self.render_input_interp_0 is None:
            self.render_input_interp_0 = embeddings
        else:
            self.render_input_interp_0 = self.get_current_frame_embeddings()
        self.render_input_interp_1 = embeddings
        self.interpolation_coef = 0

    def get_current_frame_embeddings(self):
        # Interpolate between modified latents
        frame_embedding = self.render_input_interp_0 * (1 - self.interpolation_coef) + self.render_input_interp_1 * self.interpolation_coef

        if self.sweeping_hdri:
            hdri_params = self.hdri_turntable_embeddings[self.current_hdri_embedding_frame]
            frame_embedding = self.confignet_model.set_facemodel_param_in_latents(frame_embedding,
                                                                                    "hdri_embedding",
                                                                                    hdri_params)
            self.current_hdri_embedding_frame = (self.current_hdri_embedding_frame + 1) % len(self.hdri_turntable_embeddings)

        return frame_embedding

    @staticmethod
    def print_instructions():
        print("Esc - exits the app")
        print("W,S,A,D - control the head pose")
        print("I,K,J,L - control the gaze direction")
        print("N - play a pre-set illumination sequence that rotates the light source around, pressing again ends the sequence")
        print("Z, C - change the currently driven face model parameter (attribute)")

    def drive_ui(self, key: str, test_mode=False):
        # deal with upper case
        if key >= ord("A") and key < ord("Z"):
            key += ord("a") - ord("A")
        # Escape
        if key == 27 or test_mode:
            self.exit = True

        if key == ord("a") or test_mode:
            self.rotation_offset[0, 0] -= self.rotation_angle_step_size
            print(self.rotation_offset * 180 / np.pi)
        if key == ord("d") or test_mode:
            self.rotation_offset[0, 0] += self.rotation_angle_step_size
            print(self.rotation_offset * 180 / np.pi)
        if key == ord("w") or test_mode:
            self.rotation_offset[0, 1] -= self.rotation_angle_step_size
            print(self.rotation_offset * 180 / np.pi)
        if key == ord("s") or test_mode:
            self.rotation_offset[0, 1] += self.rotation_angle_step_size
            print(self.rotation_offset * 180 / np.pi)
        if key == ord("q") or test_mode:
            self.rotation_offset[0, 2] -= self.rotation_angle_step_size
            print(self.rotation_offset * 180 / np.pi)
        if key == ord("e") or test_mode:
            self.rotation_offset[0, 2] += self.rotation_angle_step_size
            print(self.rotation_offset * 180 / np.pi)

        if key == ord("j") or test_mode:
            self.eye_rotation_offset[0, 2] -= self.rotation_angle_step_size
            print(self.eye_rotation_offset * 180 / np.pi)
        if key == ord("l") or test_mode:
            self.eye_rotation_offset[0, 2] += self.rotation_angle_step_size
            print(self.eye_rotation_offset * 180 / np.pi)
        if key == ord("i") or test_mode:
            self.eye_rotation_offset[0, 0] -= self.rotation_angle_step_size
            print(self.eye_rotation_offset * 180 / np.pi)
        if key == ord("k") or test_mode:
            self.eye_rotation_offset[0, 0] += self.rotation_angle_step_size
            print(self.eye_rotation_offset * 180 / np.pi)
        if key == ord("u") or test_mode:
            self.eye_rotation_offset[0, 1] -= self.rotation_angle_step_size
            print(self.eye_rotation_offset * 180 / np.pi)
        if key == ord("o") or test_mode:
            self.eye_rotation_offset[0, 1] += self.rotation_angle_step_size
            print(self.eye_rotation_offset * 180 / np.pi)

        if key == ord("z") or test_mode:
            self.controlled_param_idx = (self.controlled_param_idx - 1) % len(self.facemodel_param_names)
            print("Currently controlled face model parameter:", self.facemodel_param_names[self.controlled_param_idx])

        if key == ord("c") or test_mode:
            self.controlled_param_idx = (self.controlled_param_idx + 1) % len(self.facemodel_param_names)
            print("Currently controlled face model parameter:", self.facemodel_param_names[self.controlled_param_idx])

        if key == ord("n") or test_mode:
            self.sweeping_hdri = not self.sweeping_hdri
            print("Light source rotation changed to " + str(self.sweeping_hdri))

        return key
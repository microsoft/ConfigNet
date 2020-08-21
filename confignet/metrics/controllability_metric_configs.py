# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from collections import namedtuple
import inspect

ControllableAttributeConfig = namedtuple("ControllableAttributeConfig", "driven_attribute ignored_attributes facemodel_param_name facemodel_param_value facemodel_param_value_other")

class ControllabilityMetricConfigs:
    @staticmethod
    def all_configs():
        all_attributes = inspect.getmembers(ControllabilityMetricConfigs, lambda a: not inspect.isroutine(a))
        configs = [x for x in all_attributes if not (x[0].startswith('__') and x[0].endswith('__'))]

        return configs

    black_hair_config = ControllableAttributeConfig(
        driven_attribute = "Black_Hair",
        ignored_attributes = ["Blond_Hair", "Brown_Hair", "Gray_Hair"],
        facemodel_param_name = "head_hair_color",
        facemodel_param_value = (0, 1, 0),
        facemodel_param_value_other = (0, 0.1, 0.1)
    )

    blond_hair_config = ControllableAttributeConfig(
        driven_attribute = "Blond_Hair",
        ignored_attributes = ["Black_Hair", "Brown_Hair", "Gray_Hair"],
        facemodel_param_name = "head_hair_color",
        facemodel_param_value = (0, 0.1, 0.1),
        facemodel_param_value_other = (0, 1, 0)
    )

    brown_hair_config = ControllableAttributeConfig(
        driven_attribute = "Brown_Hair",
        ignored_attributes = ["Blond_Hair", "Black_Hair", "Gray_Hair"],
        facemodel_param_name = "head_hair_color",
        facemodel_param_value = (0, 0.6, 0.5),
        facemodel_param_value_other = (0, 0.1, 0.1)
    )

    gray_hair_config = ControllableAttributeConfig(
        driven_attribute = "Gray_Hair",
        ignored_attributes = ["Blond_Hair", "Brown_Hair", "Black_Hair"],
        facemodel_param_name = "head_hair_color",
        facemodel_param_value = (0.7, 0.7, 0),
        facemodel_param_value_other = (0.0, 0.7, 0)
    )

    mouth_open_config = ControllableAttributeConfig(
        driven_attribute = "Mouth_Slightly_Open",
        ignored_attributes = ["Narrow_Eyes", "Smiling"],
        facemodel_param_name = "blendshape_values",
        facemodel_param_value = {"jaw_opening": 0.2},
        facemodel_param_value_other = {"jaw_opening": -0.05}
    )

    smile_config = ControllableAttributeConfig(
        driven_attribute = "Smiling",
        ignored_attributes = ["Narrow_Eyes", "Mouth_Slightly_Open"],
        facemodel_param_name = "blendshape_values",
        facemodel_param_value = {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0},
        facemodel_param_value_other = {"mouthFrownLeft": 1.0, "mouthFrownRight": 1.0}
    )

    squint_config = ControllableAttributeConfig(
        driven_attribute = "Narrow_Eyes",
        ignored_attributes = ["Smiling", "Mouth_Slightly_Open"],
        facemodel_param_name = "blendshape_values",
        facemodel_param_value = {"EyeBLinkLeft": 0.7, "EyeBLinkRight": 0.7},
        facemodel_param_value_other = {"EyeWideLeft": 1.0, "EyeWideRight": 1.0}
    )

    mustache_config = ControllableAttributeConfig(
        driven_attribute = "Mustache",
        ignored_attributes = ["No_Beard", "Goatee", "Sideburns"],
        facemodel_param_name = "beard_style_embedding",
        # "beard_Wavy_f"
        facemodel_param_value = [
            0.8493434358437133,
            3.087059026013613,
            0.46986106722598997,
            -1.3821969829871341,
            -0.33103870587106415,
            -0.03649891754263812,
            0.049692808518749985,
            0.10727920600451613,
            -0.32365312847867017
        ],
        # "beard_none"
        facemodel_param_value_other = [
            -1.1549744366277825,
            -0.15234213575276162,
            -0.3302730721199086,
            -0.47053537289207514,
            -0.158377484760156,
            0.3357074575072504,
            -0.44934623275285585,
            0.013085621430078971,
            -0.0021044358910661896
        ]
    )

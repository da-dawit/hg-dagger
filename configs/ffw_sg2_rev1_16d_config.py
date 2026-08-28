# SPDX-FileCopyrightText: Copyright (c) 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# 16-DIM variant of ffw_sg2_rev1_config.py -- the two arms only.
#
# The 5-key version declares head [16,18), lift [18,19) and odometry [19,22) alongside the arms.
# Measured on screwing35_follower_train those six dims are unusable:
#
#   head_joint1/2         q99-q01 = 0.0     -> divide by zero
#   lift_joint            q99-q01 = 1.0e-5  -> normalised +-3
#   linear_x/y/angular_z  q99-q01 ~ 1e-3    -> normalised +-23 / +-32 / +-25 (pure sensor noise)
#   action linear_x/y/angular_z: q01 = q99 = std = 0 (regressing a constant through a zero normaliser)
#
# i.e. 6 of 22 state inputs (27%) were noise or undefined, and deployment only ever uses 16D
# (spec_sg2.MODEL_JOINTS). The dataset's meta/modality.json must be the matching 16-dim file;
# verify_dataset.py check 10 refuses any DECLARED dim whose normaliser is degenerate.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


_ACTION_KEYS = [
    "arm_left",
    "arm_right",
]


ffw_sg2_rev1_16d_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_left_head",
            "cam_left_wrist",
            "cam_right_wrist",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "arm_left",
            "arm_right",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),   #supervised horizon = 16 steps = 0.53 s at 30 fps
        modality_keys=_ACTION_KEYS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,   #action[t] = follower state[t+5]; absolute joint targets
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in _ACTION_KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.primitive_instruction"],
    ),
}


register_modality_config(ffw_sg2_rev1_16d_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

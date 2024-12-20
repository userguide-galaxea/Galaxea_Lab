# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from omni.isaac.lab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##

@configclass
class R1LiftCubeEnvCfg(joint_pos_env_cfg.R1LiftCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # # We switch here to a stiffer PD controller for IK tracking to be better.
        # self.scene.robot = GALAXEA_R1_HIGH_PD_GRIPPER_CFG.replace(
        #     prim_path="{ENV_REGEX_NS}/Robot"
        # )

        # Set actions for the specific robot type
        self.actions.left_arm_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint.*"],
            body_name="left_arm_link6",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.14]
            ),
        )

        self.actions.right_arm_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint.*"],
            body_name="right_arm_link6",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.14]
            ),
        )


@configclass
class R1LiftCubeEnvCfg_PLAY(R1LiftCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class R1LiftBinEnvCfg(joint_pos_env_cfg.R1LiftBinEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type
        self.actions.left_arm_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint.*"],
            body_name="left_arm_link6",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.14]
            ),
        )

        self.actions.right_arm_joint_pos = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint.*"],
            body_name="right_arm_link6",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.14]
            ),
        )


@configclass
class R1LiftBinEnvCfg_PLAY(R1LiftBinEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

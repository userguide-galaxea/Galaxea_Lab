# Copyright (c) 2024 Galaxea

import math

from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg


##
# Pre-defined configs
##
from omni.isaac.lab_assets import GALAXEA_R1_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class GalaxeaR1ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to GALAXEA_R1
        self.scene.table.init_state = AssetBaseCfg.InitialStateCfg(pos=(1, 0, -1.05), rot=(0.70711, 0.0, 0.0, 0.70711))
        self.scene.robot = GALAXEA_R1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0, 0, 0)
        # override rewards
        self.rewards.right_end_effector_position_tracking.params["asset_cfg"].body_names = ["right_gripper_2_Link"]
        self.rewards.right_end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["right_gripper_2_Link"]
        self.rewards.right_end_effector_orientation_tracking.params["asset_cfg"].body_names = ["right_gripper_2_Link"]
        self.rewards.left_end_effector_position_tracking.params["asset_cfg"].body_names = ["left_gripper_1_Link"]
        self.rewards.left_end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["left_gripper_1_Link"]
        self.rewards.left_end_effector_orientation_tracking.params["asset_cfg"].body_names = ["left_gripper_1_Link"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["(left|right)_arm_joint[1-6]"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose_right.body_name = "right_gripper_2_Link"
        self.commands.ee_pose_right.ranges.pitch = (0, 0)
        self.commands.ee_pose_right.ranges.pitch = (-0.5*math.pi, 0.5*math.pi)
        self.commands.ee_pose_left.body_name = "left_gripper_1_Link"
        self.commands.ee_pose_left.ranges.pitch = (0, 0)
        self.commands.ee_pose_left.ranges.pitch = (-0.5*math.pi, 0.5*math.pi)


@configclass
class GalaxeaR1ReachEnvCfg_PLAY(GalaxeaR1ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 1.
        # disable randomization for play
        self.observations.policy.enable_corruption = False
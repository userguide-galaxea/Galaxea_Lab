# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/galaxea/basic/ik_control.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on using the differential IK controller."
)
parser.add_argument("--robot", type=str, default="R1", help="Name of the robot.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from omni.isaac.lab_assets import (
    GALAXEA_R1_HIGH_PD_CFG,
    GALAXEA_R1_HIGH_PD_GRIPPER_CFG,
)  # isort:skip


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75)),
    )

    # articulation
    if args_cli.robot == "R1":
        robot = GALAXEA_R1_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "R1StrongGripper":
        robot = GALAXEA_R1_HIGH_PD_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(
            f"Robot {args_cli.robot} is not supported. Valid: R1, R1StrongGripper"
        )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, num_envs=scene.num_envs, device=sim.device
    )

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    left_ee_marker = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_current/left")
    )
    left_goal_marker = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_goal/left")
    )

    # Define goals for the arm
    # init pose: 0.3864, 0.5237, 1.1475, 9.6247e-05,  9.7698e-01, -2.1335e-01,  3.9177e-04
    left_ee_goals = [
        [0.4, 0.3, 1.0, 0.0, 0.707, 0.707, 0],
        [0.2, 0.4, 1.25, 0.707, 0.707, 0.0, 0.0],
        [0.3, 0.2, 1.15, 0.0, 1.0, 0.0, 0.0],
    ]
    left_ee_goals = torch.tensor(left_ee_goals, device=sim.device)

    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    left_ik_commands = torch.zeros(
        scene.num_envs, diff_ik_controller.action_dim, device=robot.device
    )
    left_ik_commands[:] = left_ee_goals[current_goal_idx]

    # Specify robot-specific parameters
    left_arm_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["left_arm_joint.*"], body_names=["left_arm_link6"]
    )
    right_arm_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["right_arm_joint.*"], body_names=["right_arm_link6"]
    )

    # Resolving the scene entities
    left_arm_entity_cfg.resolve(scene)
    right_arm_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    print("robot.is_fixed_base: ", robot.is_fixed_base)
    if robot.is_fixed_base:
        left_ee_jacobi_idx = left_arm_entity_cfg.body_ids[0] - 1
        right_ee_jacobi_idx = right_arm_entity_cfg.body_ids[0] - 1
    else:
        left_ee_jacobi_idx = left_arm_entity_cfg.body_ids[0]
        right_ee_jacobi_idx = right_arm_entity_cfg.body_ids[0]

    left_arm_joint_ids = left_arm_entity_cfg.joint_ids
    right_arm_joint_ids = right_arm_entity_cfg.joint_ids
    num_arm_joints = len(left_arm_joint_ids)
    if num_arm_joints != len(right_arm_joint_ids):
        raise ValueError("The number of left and right arm joints should be the same.")

    left_gripper_entity_cfg = SceneEntityCfg("robot", joint_names=["left_gripper_.*"])
    right_gripper_entity_cfg = SceneEntityCfg("robot", joint_names=["right_gripper_.*"])
    left_gripper_entity_cfg.resolve(scene)
    right_gripper_entity_cfg.resolve(scene)

    # get left/right gripper joint ids
    left_gripper_joint_ids = left_gripper_entity_cfg.joint_ids
    right_gripper_joint_ids = right_gripper_entity_cfg.joint_ids
    num_gripper_joints = len(left_gripper_joint_ids)
    if num_gripper_joints != len(right_gripper_joint_ids):
        raise ValueError(
            "The number of left and right gripper joints should be the same."
        )

    print("-------------------------------------------------")
    print("left body_ids: ", left_arm_entity_cfg.body_ids)
    print("left joint_ids: ", left_arm_joint_ids)
    print("left_ee_jacobi_idx: ", left_ee_jacobi_idx)
    print("right body_ids: ", right_arm_entity_cfg.body_ids)
    print("right joint_ids: ", right_arm_joint_ids)
    print("right_ee_jacobi_idx: ", right_ee_jacobi_idx)
    print("num_arm_joints: ", num_arm_joints)
    print("*************************************************")
    print("left gripper joint_ids: ", left_gripper_joint_ids)
    print("right gripper joint_ids: ", right_gripper_joint_ids)
    print("num gripper joints: ", num_gripper_joints)
    print("-------------------------------------------------")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            left_ik_commands[:] = left_ee_goals[current_goal_idx]
            left_joint_pos_des = joint_pos[:, left_arm_entity_cfg.joint_ids].clone()
            left_gripper_joint_pos_des = joint_pos[:, left_gripper_joint_ids].clone()
            # reset controller
            # diff_ik_controller.reset()
            diff_ik_controller.set_command(left_ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(left_ee_goals)
        else:
            # obtain quantities from simulation
            left_jacobian = robot.root_physx_view.get_jacobians()[
                :, left_ee_jacobi_idx, :, left_arm_entity_cfg.joint_ids
            ]
            left_ee_pose_w = robot.data.body_state_w[
                :, left_arm_entity_cfg.body_ids[0], 0:7
            ]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            left_joint_pos = robot.data.joint_pos[:, left_arm_entity_cfg.joint_ids]
            # compute frame in root frame
            left_ee_pos_b, left_ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                left_ee_pose_w[:, 0:3],
                left_ee_pose_w[:, 3:7],
            )
            # compute the joint commands
            left_joint_pos_des = diff_ik_controller.compute(
                left_ee_pos_b, left_ee_quat_b, left_jacobian, left_joint_pos
            )

        left_gripper_joint_pos_des[:] = torch.full(
            (num_gripper_joints,), 0.03 if count < 100 else 0.0, device=robot.device
        )

        # apply actions
        robot.set_joint_position_target(
            left_joint_pos_des, joint_ids=left_arm_entity_cfg.joint_ids
        )
        # print("left_gripper_joint_pos_des: ", left_gripper_joint_pos_des)
        robot.set_joint_position_target(
            left_gripper_joint_pos_des, joint_ids=left_gripper_joint_ids
        )
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        left_ee_pose_w = robot.data.body_state_w[
            :, left_arm_entity_cfg.body_ids[0], 0:7
        ]
        # update marker positions
        left_ee_marker.visualize(left_ee_pose_w[:, 0:3], left_ee_pose_w[:, 3:7])
        left_goal_marker.visualize(
            left_ik_commands[:, 0:3] + scene.env_origins, left_ik_commands[:, 3:7]
        )


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

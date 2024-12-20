# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Keyboard teleoperation for Isaac Lab environments."
)
parser.add_argument(
    "--cpu", action="store_true", default=False, help="Use CPU pipeline."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--device",
    type=str,
    default="keyboard",
    help="Device for interacting with environment",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--sensitivity", type=float, default=1.0, help="Sensitivity factor."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch

import carb

from omni.isaac.lab.devices import Se3KeyboardBiman

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG


def pre_process_actions(
    delta_pose: torch.Tensor, gripper_command: bool
) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
        gripper_vel[:] = -1.0 if gripper_command else 1.0
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        use_gpu=not args_cli.cpu,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # modify configuration
    env_cfg.terminations.time_out = None

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        carb.log_warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    if args_cli.device.lower() == "keyboard":
        teleop_interface = Se3KeyboardBiman(
            pos_sensitivity=0.05 * args_cli.sensitivity,
            rot_sensitivity=0.1 * args_cli.sensitivity,
        )
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.device}'. Supported: 'keyboard'."
        )
    # add teleoperation key for env reset
    teleop_interface.add_callback("R", env.reset)
    # print helper for keyboard
    print(teleop_interface)

    # add object marker
    object_marker_cfg = FRAME_MARKER_CFG.copy()
    object_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    object_marker_cfg.prim_path = "/Visuals/FrameTransformer/Object"
    object_marker = VisualizationMarkers(object_marker_cfg)
    object = env.unwrapped.scene["object"]

    # reset environment
    env.reset()
    teleop_interface.reset()

    left_ee_frame = env.unwrapped.scene["left_ee_frame"]
    right_ee_frame = env.unwrapped.scene["right_ee_frame"]
    print(left_ee_frame.data.target_frame_names)
    print(right_ee_frame.data.target_frame_names)

    # simulate environment
    count = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # print(object.data.root_state_w[:, 0:7])
            object_marker.visualize(
                object.data.root_state_w[:, 0:3],
                object.data.root_state_w[:, 3:7],
            )
            # get keyboard command
            left_delta_pose, left_gripper_cmd, right_delta_pose, right_gripper_cmd = (
                teleop_interface.advance()
            )

            # convert to torch
            left_delta_pose = left_delta_pose.astype("float32")
            left_delta_pose = torch.tensor(
                left_delta_pose, device=env.unwrapped.device
            ).repeat(env.unwrapped.num_envs, 1)

            right_delta_pose = right_delta_pose.astype("float32")
            right_delta_pose = torch.tensor(
                right_delta_pose, device=env.unwrapped.device
            ).repeat(env.unwrapped.num_envs, 1)
            # print(
            #     "left_delta_pose: {}, left_gripper_action: {}".format(
            #         left_delta_pose, left_gripper_cmd
            #     )
            # )
            # print(
            #     "right_delta_pose: {}, right_gripper_action: {}".format(
            #         right_delta_pose, right_gripper_cmd
            #     )
            # )

            # pre-process actions
            left_action = pre_process_actions(left_delta_pose, left_gripper_cmd)
            right_action = pre_process_actions(right_delta_pose, right_gripper_cmd)

            left_arm_action = left_action[:, :-1]
            left_gripper_action = left_action[:, -1].unsqueeze(1)

            right_arm_action = right_action[:, :-1]
            right_gripper_action = right_action[:, -1].unsqueeze(1)
            # print(
            #     "left_arm_action:l {}, left_gripper_action: {}".format(
            #         left_arm_action, left_gripper_action
            #     )
            # )
            # print(
            #     "right_arm_action: {}, right_gripper_action: {}".format(
            #         right_arm_action, right_gripper_action
            #     )
            # )
            if count < 3:
                print("count: ", count)
                print("left_ee_pos: ", left_ee_frame.data.target_pos_w)
                print("left_ee_quat: ", left_ee_frame.data.target_quat_w)
                print("right_ee_pos: ", right_ee_frame.data.target_pos_w)
                print("right_ee_quat: ", right_ee_frame.data.target_quat_w)

            actions = torch.concat(
                [
                    left_arm_action,
                    left_gripper_action,
                    right_arm_action,
                    right_gripper_action,
                ],
                dim=1,
            )
            # print("actions: ", actions)
            # apply actions
            env.step(actions)
            count += 1
            # print("obs: ", obs["policy"][0])

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

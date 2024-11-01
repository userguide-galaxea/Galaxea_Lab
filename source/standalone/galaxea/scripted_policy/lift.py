# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p source/standalone/galaxea/state_machine/lift_bin_direct.py --num_envs 10

"""

"""Launch Omniverse Toolkit first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Pick and lift state machine for lift environments."
)
parser.add_argument(
    "--cpu", action="store_true", default=False, help="Use CPU pipeline."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-R1-Lift-Cube-IK-Abs-Direct-v0",
    help="Name of the task.",
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
    "--save_data", action="store_true", default=False, help="save data or not"
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch

from omni.isaac.lab.assets.rigid_object.rigid_object_data import RigidObjectData
from omni.isaac.lab.utils.math import (
    combine_frame_transforms,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
)

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from state_machine.pick_n_lift import PickAndLiftSm


def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        use_gpu=not args_cli.cpu,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # reset environment at start
    obs, _ = env.reset()

    # create state machine
    sm = PickAndLiftSm(
        args_cli.task,
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
    )

    count = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get goal position
            goal_pos = obs["policy"]["goal_pose"][..., :3]
            # generate actions
            actions = sm.generate_actions(env, goal_pos)

            # step environment
            obs, reward, terminated, truncated, info = env.step(actions)

            # reset state machine
            if truncated.any() or terminated.any():
                dones = truncated | terminated
                reset_idx = dones.nonzero(as_tuple=False).squeeze(-1)
                # print(
                #     "count: {}\nterminated: {}\ntruncated: {}\ndone: {}\nreset_idx:{}".format(
                #         count, terminated, truncated, dones, reset_idx
                #     )
                # )
                sm.reset_idx(reset_idx)

            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

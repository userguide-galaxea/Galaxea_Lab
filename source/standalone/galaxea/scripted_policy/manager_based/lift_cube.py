# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p source/standalone/galaxea/state_machine/lift_box.py --num_envs 32

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
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
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
from collections.abc import Sequence

import warp as wp


from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.assets.rigid_object.rigid_object_data import RigidObjectData
from omni.isaac.lab.utils.math import (
    combine_frame_transforms,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
)

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.galaxea.manager_based.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab_tasks.galaxea.manager_based.lift.config.agents.init_pose import (
    LEFT_EE_POSE,
    RIGHT_EE_POSE,
)
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(1.5)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        # des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.GRASP_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        # des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full(
            (self.num_envs,), 0, dtype=torch.int32, device=self.device
        )
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)
        # print("[DEBUG] offset: ", self.offset)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(
        self,
        ee_pose: torch.Tensor,
        object_pose: torch.Tensor,
        des_object_pose: torch.Tensor,
    ):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # pose_offset = torch.zeros_like(object_pose)
        # orientation_offset_euler = torch.zeros_like(object_pose[:, :3])
        # # orientation_offset_euler[:, 0] =  3.1415926
        # orientation_offset_euler[:, 0] = 1.5707963
        # print("orientation_offset_euler: ", orientation_offset_euler)
        # orientation_offset_quat = quat_from_euler_xyz(
        #     orientation_offset_euler[:, 0],
        #     orientation_offset_euler[:, 1],
        #     orientation_offset_euler[:, 2],
        # )  # in w, x, y, z
        # pose_offset[:, 3:7] = orientation_offset_quat
        # print("pose_offset: ", pose_offset)
        # processed_object_position, processed_object_orientation = (
        #     combine_frame_transforms(
        #         object_pose[:, :3],
        #         object_pose[:, 3:7],
        #         pose_offset[:, :3],
        #         pose_offset[:, 3:7],
        #     )
        # )
        # processed_object_pose = torch.cat(
        #     [processed_object_position, processed_object_orientation], dim=-1
        # )
        # print("object_pose: ", object_pose)
        # print("processed_object_pose: ", processed_object_pose)

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # print("des_ee_pose: ", des_ee_pose)
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


class PoseMarker:
    # -- add object marker
    object_marker_cfg = FRAME_MARKER_CFG.copy()
    object_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    object_marker_cfg.prim_path = "/Visuals/FrameTransformer/Object"
    object_marker = VisualizationMarkers(object_marker_cfg)

    left_object_marker_cfg = FRAME_MARKER_CFG.copy()
    left_object_marker_cfg.markers["frame"].scale = (0.075, 0.075, 0.075)
    left_object_marker_cfg.prim_path = "/Visuals/FrameTransformer/Object_Left"
    left_object_marker = VisualizationMarkers(left_object_marker_cfg)

    right_object_marker_cfg = FRAME_MARKER_CFG.copy()
    right_object_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    right_object_marker_cfg.prim_path = "/Visuals/FrameTransformer/Object_Right"
    right_object_marker = VisualizationMarkers(right_object_marker_cfg)

    left_action_marker_cfg = FRAME_MARKER_CFG.copy()
    left_action_marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
    left_action_marker_cfg.prim_path = "/Visuals/FrameTransformer/Action_Left"
    left_action_marker = VisualizationMarkers(left_action_marker_cfg)

    right_action_marker_cfg = FRAME_MARKER_CFG.copy()
    right_action_marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
    right_action_marker_cfg.prim_path = "/Visuals/FrameTransformer/Action_Right"
    right_action_marker = VisualizationMarkers(right_action_marker_cfg)


def main():
    # parse configuration
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-R1-IK-Abs-v0",
        use_gpu=not args_cli.cpu,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("Isaac-Lift-Cube-R1-IK-Abs-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()

    # create action buffers (position + quaternion)
    # print("action_shape: ", env.unwrapped.action_space.shape)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)

    # -- object frame
    object_data: RigidObjectData = env.unwrapped.scene["object"].data
    object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins

    # -- target object frame
    desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros(
        (env.unwrapped.num_envs, 4), device=env.unwrapped.device
    )
    desired_orientation[:, 1] = 1.0

    # create state machine
    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
    )

    # get initial end-effector pose
    left_ee_frame = env.unwrapped.scene["left_ee_frame"]
    left_ee_pose = left_ee_frame.data.target_pos_w[..., 0, :].clone()
    left_ee_pose[:, 0:3] = LEFT_EE_POSE[0:3]
    left_tcp_rest_position = left_ee_pose - env.unwrapped.scene.env_origins
    left_tcp_rest_orientation = left_ee_frame.data.target_quat_w[..., 0, :].clone()
    left_tcp_rest_orientation[:, 0:] = LEFT_EE_POSE[3:]
    # print(
    #     "left_tcp_rest position: {}, orientation: {}".format(
    #         left_tcp_rest_position, left_tcp_rest_orientation
    #     )
    # )

    right_ee_frame = env.unwrapped.scene["right_ee_frame"]
    right_ee_pose = right_ee_frame.data.target_pos_w[..., 0, :].clone()
    right_ee_pose[:, 0:3] = RIGHT_EE_POSE[0:3]
    right_tcp_rest_position = right_ee_pose - env.unwrapped.scene.env_origins
    right_tcp_rest_orientation = right_ee_frame.data.target_quat_w[..., 0, :].clone()
    right_tcp_rest_orientation[:, 0:] = RIGHT_EE_POSE[3:]
    # print(
    #     "right_tcp_rest position: {}, orientation: {}".format(
    #         right_tcp_rest_position, right_tcp_rest_orientation
    #     )
    # )
    actions = torch.cat(
        [
            left_tcp_rest_position,
            left_tcp_rest_orientation,
            pick_sm.des_gripper_state.unsqueeze(-1),
            right_tcp_rest_position,
            right_tcp_rest_orientation,
            pick_sm.des_gripper_state.unsqueeze(-1),
        ],
        dim=-1,
    )
    actions[:, 7] = 1.0
    actions[:, -1] = 1.0
    left_actions = actions[:, 0:8]
    right_actions = actions[:, 8:]
    # print(
    #     "Initial actions: {}, \nleft_actions: {}, \nright_actions: {}".format(
    #         actions, left_actions, right_actions
    #     )
    # )

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            # print("[start] output actions: ", actions)
            dones = env.step(actions)[-2]

            # observations
            # -- end-effector frame
            left_ee_frame = env.unwrapped.scene["left_ee_frame"]
            left_tcp_rest_position = (
                left_ee_frame.data.target_pos_w[..., 0, :].clone()
                - env.unwrapped.scene.env_origins
            )
            left_tcp_rest_orientation = left_ee_frame.data.target_quat_w[
                ..., 0, :
            ].clone()

            right_ee_frame = env.unwrapped.scene["right_ee_frame"]
            right_tcp_rest_position = (
                right_ee_frame.data.target_pos_w[..., 0, :].clone()
                - env.unwrapped.scene.env_origins
            )
            right_tcp_rest_orientation = right_ee_frame.data.target_quat_w[
                ..., 0, :
            ].clone()

            # -- object frame
            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            object_orientation = object_data.root_quat_w
            object_orientation_yaw = euler_xyz_from_quat(object_orientation)[2]
            # object_orientation_yaw = (object_orientation_yaw + torch.pi) % (
            #     2 * torch.pi
            # ) - torch.pi
            # print("object_orientation yaw: {}".format(object_orientation_yaw))
            # -- target object frame
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[
                ..., :3
            ]
            object_position_vis = object_position + env.unwrapped.scene.env_origins
            PoseMarker.object_marker.visualize(object_position_vis, object_orientation)

            # the grasping poses on object for left/right grippers
            left_object_offset = torch.zeros_like(object_position)
            left_object_offset[:, 1] = 0.15
            left_object_position, _ = combine_frame_transforms(
                object_position, object_orientation, left_object_offset
            )
            left_object_position_vis = (
                left_object_position + env.unwrapped.scene.env_origins
            )
            PoseMarker.left_object_marker.visualize(
                left_object_position_vis, object_orientation
            )

            right_object_offset = torch.zeros_like(object_position)
            right_object_offset[:, 1] = -0.15
            right_object_position, _ = combine_frame_transforms(
                object_position, object_orientation, right_object_offset
            )
            right_object_position_vis = (
                right_object_position + env.unwrapped.scene.env_origins
            )
            PoseMarker.right_object_marker.visualize(
                right_object_position_vis, object_orientation
            )

            # -- update right desired position
            right_desired_position = desired_position.clone()
            right_desired_position[:, 1] -= 0.15
            # -- update right desired orientation
            r_orientation_offset_euler = torch.zeros_like(right_object_position)
            r_orientation_offset_euler[:, 0] = 1.5707963 + object_orientation_yaw
            r_orientation_offset_quat = quat_from_euler_xyz(
                r_orientation_offset_euler[:, 0],
                r_orientation_offset_euler[:, 1],
                r_orientation_offset_euler[:, 2],
            )
            # conver to (z, w, x, y)
            right_desired_orientation = r_orientation_offset_quat[:, [3, 0, 1, 2]]
            # print(
            #     "right desired_pose: {}, {}".format(
            #         right_desired_position, right_desired_orientation
            #     )
            # )

            # -- update left desired position
            left_desired_position = desired_position.clone()
            left_desired_position[:, 1] += 0.15
            # -- update right desired orientation
            l_orientation_offset_euler = torch.zeros_like(left_object_position)
            l_orientation_offset_euler[:, 0] = -1.5707963 + object_orientation_yaw
            l_orientation_offset_quat = quat_from_euler_xyz(
                l_orientation_offset_euler[:, 0],
                l_orientation_offset_euler[:, 1],
                l_orientation_offset_euler[:, 2],
            )
            # conver to (z, w, x, y)
            left_desired_orientation = l_orientation_offset_quat[:, [3, 0, 1, 2]]
            # print(
            #     "left desired_pose: {}, {}".format(
            #         left_desired_position, left_desired_orientation
            #     )
            # )

            # advance state machine
            left_actions = pick_sm.compute(
                torch.cat([left_tcp_rest_position, left_tcp_rest_orientation], dim=-1),
                torch.cat([left_object_position, left_desired_orientation], dim=-1),
                torch.cat([left_desired_position, left_desired_orientation], dim=-1),
            )
            PoseMarker.left_action_marker.visualize(
                left_actions[:, :3] + env.unwrapped.scene.env_origins,
                left_actions[:, 3:7],
            )

            right_actions = pick_sm.compute(
                torch.cat(
                    [right_tcp_rest_position, right_tcp_rest_orientation], dim=-1
                ),
                torch.cat([right_object_position, right_desired_orientation], dim=-1),
                torch.cat([right_desired_position, right_desired_orientation], dim=-1),
            )
            PoseMarker.right_action_marker.visualize(
                right_actions[:, :3] + env.unwrapped.scene.env_origins,
                right_actions[:, 3:7],
            )

            # print("[DEBUG] left_actions: {}", left_actions)
            # print("[DEBUG] right_actions: {}", right_actions)
            actions = torch.cat([left_actions, right_actions], dim=-1)

            # # reset state machine
            if dones.any():
                # print("[DEBUG] lift cube done...")
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

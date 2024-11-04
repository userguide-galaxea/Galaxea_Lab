import torch
import warp as wp
import gymnasium as gym
import copy
from collections.abc import Sequence
from omni.isaac.lab.utils.math import (
    combine_frame_transforms,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
)


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
    APPROACH_WITH_OBJECT = wp.constant(5)
    PUT_DOWN_OBJECT = wp.constant(6)
    RELEASE_OBJECT = wp.constant(7)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.5)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(0.5)
    APPROACH_WITH_OBJECT = wp.constant(0.5)
    PUT_DOWN_OBJECT = wp.constant(0.5)
    RELEASE_OBJECT = wp.constant(0.5)


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    move_object_pose: wp.array(dtype=wp.transform),
    put_down_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    state = sm_state[tid]
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            sm_state[tid] = PickSmState.APPROACH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            sm_state[tid] = PickSmState.GRASP_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
            sm_state[tid] = PickSmState.APPROACH_WITH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_WITH_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], move_object_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_WITH_OBJECT:
            sm_state[tid] = PickSmState.PUT_DOWN_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.PUT_DOWN_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], put_down_object_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickSmWaitTime.PUT_DOWN_OBJECT:
            sm_state[tid] = PickSmState.RELEASE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.RELEASE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], move_object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.RELEASE_OBJECT:
            sm_state[tid] = PickSmState.RELEASE_OBJECT
            sm_wait_time[tid] = 0.0        
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]

class PickAndPlaceRightArmSm:
    """A variaty of PickAndLiftSm, using right arm only.
    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:
    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose1.
    6. APPROACH_WITH_OBJECT: The robot moves to the desired pose2 with the object.
    7. PUT_DOWN_OBJECT: The robot puts down the object.
    8. RELEASE_OBJECT: The robot releases the object.
    """
    def __init__(
        self, task: str, dt: float, num_envs: int, device: torch.device | str = "cpu"
    ):
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
        self.task = task
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
        self.set_task_config()
        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)
    def set_task_config(self):
        # offsets for different tasks
        # x, y, z, roll, pitch, yaw
        self.task_offset = torch.zeros(6, device=self.device)
        self.task_offset[1] = 0.15
        self.task_offset[2] = -0.1
        self.task_offset[5] = 1.5707963
        print("task: {}, offset: {}".format(self.task, self.task_offset))
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
        move_object_pose: torch.Tensor,
    ):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        move_object_pose = move_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        put_down_object_pose = copy.deepcopy(move_object_pose)
        put_down_object_pose[:, 2] = 0.9
        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)
        move_object_pose_wp = wp.from_torch(move_object_pose.contiguous(), wp.transform)
        put_down_object_pose_wp = wp.from_torch(put_down_object_pose.contiguous(), wp.transform)
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
                move_object_pose_wp,
                put_down_object_pose_wp,
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
    def generate_actions(self, env: gym.Env, goal_pos: torch.Tensor) -> torch.Tensor:
        self.object_id = env.unwrapped.object_id
        # -- end-effector frame
        left_ee_frame = env.unwrapped.scene["left_ee_frame"]
        left_tcp_rest_position = (
            left_ee_frame.data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
        )
        left_tcp_rest_orientation = left_ee_frame.data.target_quat_w[..., 0, :]
        right_ee_frame = env.unwrapped.scene["right_ee_frame"]
        right_tcp_rest_position = (
            right_ee_frame.data.target_pos_w[..., 0, :]
            - env.unwrapped.scene.env_origins
        )
        right_tcp_rest_orientation = right_ee_frame.data.target_quat_w[..., 0, :]
        # -- object frame
        # object_data = env.unwrapped.scene["object"].data
        object_data = env.unwrapped.scene[f"object{self.object_id}"].data
        object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
        object_orientation = object_data.root_quat_w
        object_orientation_yaw = euler_xyz_from_quat(object_orientation)[2]
        # -- the grasping poses on object for right grippers
        right_object_offset = torch.zeros_like(object_position)
        # single hand just grasp at the center
        # right_object_offset[:, 1] = -self.task_offset[1]
        right_object_offset[:, 2] = -0.1
        right_object_position, _ = combine_frame_transforms(
            object_position, object_orientation, right_object_offset
        )
        # -- update right desired position
        right_desired_position = goal_pos.clone()
        # single hand just grasp at the center
        # right_desired_position[:, 1] -= self.task_offset[1]
        # -- update right desired orientation
        r_orientation_offset_euler = torch.zeros_like(right_object_position)
        r_orientation_offset_euler[:, 0] = object_orientation_yaw #- self.task_offset[5]
        r_orientation_offset_quat = quat_from_euler_xyz(
            r_orientation_offset_euler[:, 0],
            r_orientation_offset_euler[:, 1],
            r_orientation_offset_euler[:, 2],
        )
        # conver to (z, w, x, y)
        right_desired_orientation = r_orientation_offset_quat[:, [3, 0, 1, 2]]

        num_rows = left_tcp_rest_position.size(0)

        additional_tensor = torch.ones((num_rows, 1), device=self.device)
        # left hand always stay at the same position
        left_actions = torch.cat([left_tcp_rest_position, left_tcp_rest_orientation, additional_tensor], dim=-1)
            
        right_move_position_data = env.unwrapped.scene[f"object3"].data
        right_move_position_pos = right_move_position_data.root_pos_w - env.unwrapped.scene.env_origins
        right_move_position_pos[0, 2] = goal_pos[0, 2]
        right_move_orientation = right_move_position_data.root_quat_w
        # -- the grasping poses on object for right grippers
        right_move_object_offset = torch.zeros_like(right_move_position_pos)
        # single hand just grasp at the center
        right_move_object_offset[:, 0] = -0.1
        # right_object_offset[:, 2] = -self.task_offset[2]
        right_move_position_pos, _ = combine_frame_transforms(
            right_move_position_pos, right_move_orientation, right_move_object_offset
        )

        right_move_orientation = right_move_position_data.root_quat_w
        r_move_orientation_offset_euler = torch.zeros_like(right_move_position_pos)
        r_move_orientation_offset_euler[:, 0] = euler_xyz_from_quat(right_move_orientation)[2] - self.task_offset[5]
        r_move_orientation_offset_quat = quat_from_euler_xyz(
            r_move_orientation_offset_euler[:, 0],
            r_move_orientation_offset_euler[:, 1],
            r_move_orientation_offset_euler[:, 2],
        )

        # convert to (z, w, x, y)
        right_move_orientation = r_move_orientation_offset_quat[:, [3, 0, 1, 2]]
        right_actions = self.compute(
            torch.cat([right_tcp_rest_position, right_tcp_rest_orientation], dim=-1),
            torch.cat([right_object_position, right_desired_orientation], dim=-1),
            torch.cat([right_desired_position, right_desired_orientation], dim=-1),
            torch.cat([right_move_position_pos, right_move_orientation], dim=-1),
        )
        # right_actions = torch.cat([right_tcp_rest_position, right_tcp_rest_orientation, torch.tensor([[1.0]], device=self.device)], dim=-1)
        actions = torch.cat([left_actions, right_actions], dim=-1)
        return actions
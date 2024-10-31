from __future__ import annotations

import torch

from omni.isaac.core.prims import XFormPrimView
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject, AssetBase
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.utils.math import (
    sample_uniform,
    combine_frame_transforms,
    subtract_frame_transforms,
    quat_from_euler_xyz,
    quat_mul,
    skew_symmetric_matrix,
    matrix_from_quat,
)
from omni.isaac.lab.sensors.frame_transformer import FrameTransformer
from omni.isaac.lab.sensors import Camera
from omni.isaac.lab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)

from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .lift_env_cfg import (
    # R1LiftCubeEnvCfg,
    R1LiftCubeAbsEnvCfg,
    R1LiftCubeRelEnvCfg,
    # R1LiftBinEnvCfg,
    R1LiftBinAbsEnvCfg,
    R1LiftBinRelEnvCfg,
)


class R1LiftEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_observations()
    #   |-- _get_rewards()
    #   |-- _get_dones()
    #   |-- _reset_idx(env_ids)

    cfg: (
        # R1LiftBinEnvCfg
        # | R1LiftCubeEnvCfg
        R1LiftCubeAbsEnvCfg
        | R1LiftCubeRelEnvCfg
        | R1LiftBinAbsEnvCfg
        | R1LiftBinRelEnvCfg
    )
    """ Lift an object to a target position.
    """

    def __init__(
        self,
        cfg: (
            # R1LiftBinEnvCfg
            # | R1LiftCubeEnvCfg
            R1LiftCubeAbsEnvCfg
            | R1LiftCubeRelEnvCfg
            | R1LiftBinAbsEnvCfg
            | R1LiftBinRelEnvCfg
        ),
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # buffer for joint position targets
        # target joints == actions, the results of _process_action()

        # action type
        self.action_type = self.cfg.action_type

        # joint limits
        self.robot_joint_lower_limits = self._robot.data.soft_joint_pos_limits[
            0, :, 0
        ].to(device=self.device)
        self.robot_joint_upper_limits = self._robot.data.soft_joint_pos_limits[
            0, :, 1
        ].to(device=self.device)

        # track goal reset state
        self.reset_goal_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # default goal pose, i.e. the init target pose in the robot base_link frame
        self.goal_rot = torch.zeros(
            (self.num_envs, 4), dtype=torch.float, device=self.device
        )
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )
        self.goal_pos[:, :] = torch.tensor([0.4, 0.0, 1.2], device=self.device)

        # vis markers
        self.vis_markers = VisualizationMarkers(self.cfg.vis_markers_cfg)
        self.marker_indices = [
            i
            for i in range(self.vis_markers.num_prototypes)
            for _ in range(self.scene.num_envs)
        ]
        self.set_debug_vis(self.cfg.debug_vis)

        # end-effector offset w.r.t the *_arm_link6 frame
        self.ee_offset_pos = torch.tensor([0.0, 0.0, 0.15], device=self.device).repeat(
            self.num_envs, 1
        )
        self.ee_offset_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)

        # left/right arm/gripper joint ids
        self._setup_robot()
        # ik controller
        self._setup_ik_controller()

        self.succ = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.object_id = 0

        print("R1LiftEnv is initialized. ActionType: ", self.action_type)

    def _setup_scene(self):
        # add robot, object
        self._robot = Articulation(self.cfg.robot_cfg)
        self._object = RigidObject(self.cfg.object_cfg)

        # add table which is a static object
        if self.cfg.table_cfg.spawn is not None:
            self.cfg.table_cfg.spawn.func(
                self.cfg.table_cfg.prim_path,
                self.cfg.table_cfg.spawn,
                translation=self.cfg.table_cfg.init_state.pos,
                orientation=self.cfg.table_cfg.init_state.rot,
            )
        # self._table = RigidObject(self.cfg.table_cfg)

        # add camera
        if self.cfg.enable_camera:
            self._front_camera = Camera(self.cfg.front_camera_cfg)
            self._left_wrist_camera = Camera(self.cfg.left_wrist_camera_cfg)
            self._right_wrist_camera = Camera(self.cfg.right_wrist_camera_cfg)
            self.scene.sensors["front_camera"] = self._front_camera
            self.scene.sensors["left_wrist_camera"] = self._left_wrist_camera
            self.scene.sensors["right_wrist_camera"] = self._right_wrist_camera

        # frame transformer
        self._left_ee_frame = FrameTransformer(self.cfg.left_ee_frame_cfg)
        self._right_ee_frame = FrameTransformer(self.cfg.right_ee_frame_cfg)

        # add to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object
        self.scene.sensors["left_ee_frame"] = self._left_ee_frame
        self.scene.sensors["right_ee_frame"] = self._right_ee_frame
        self.scene.extras["table"] = XFormPrimView(
            self.cfg.table_cfg.prim_path, reset_xform_properties=False
        )
        # self.scene.rigid_objects["table"] = self._table

        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(color=(1.0, 1.0, 1.0))
        )
        # self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        # self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        print("Scene is set up.")

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self._process_action(actions)

    def _apply_action(self):
        # set left/right arm/gripper joint position targets
        self._robot.set_joint_position_target(
            self.left_arm_joint_pos_target, self.left_arm_joint_ids
        )
        self._robot.set_joint_position_target(
            self.left_gripper_joint_pos_target, self.left_gripper_joint_ids
        )
        self._robot.set_joint_position_target(
            self.right_arm_joint_pos_target, self.right_arm_joint_ids
        )
        self._robot.set_joint_position_target(
            self.right_gripper_joint_pos_target, self.right_gripper_joint_ids
        )


    # post-physics step calls
    def _get_observations(self) -> dict:
        # note: the position in observations should in the local frame

        # get robot end-effector pose
        left_ee_pos = (
            self._left_ee_frame.data.target_pos_w[..., 0, :] - self.scene.env_origins
        )
        right_ee_pos = (
            self._right_ee_frame.data.target_pos_w[..., 0, :] - self.scene.env_origins
        )
        left_ee_pose = torch.cat(
            [left_ee_pos, self._left_ee_frame.data.target_quat_w[..., 0, :]], dim=-1
        )
        right_ee_pose = torch.cat(
            [right_ee_pos, self._right_ee_frame.data.target_quat_w[..., 0, :]], dim=-1
        )

        # get robot joint position
        # joint_position_scaled = (
        #     2.0
        #     * (self._robot.data.joint_pos - self.robot_joint_lower_limits)
        #     / (self.robot_joint_upper_limits - self.robot_joint_lower_limits)
        #     - 1.0
        # )
        joint_pos, joint_vel = self._process_joint_value()
        # print("[obs] returned joint_pos:{}\n\t  joint_vel: {}".format(joint_pos, joint_vel))

        # get object pose
        object_pos = self._object.data.root_pos_w - self.scene.env_origins
        object_pose = torch.cat([object_pos, self._object.data.root_quat_w], dim=-1)

        obs = {
            # robot joint position: dim=(6+2)*2
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            # robot ee pose: dim=7*2
            "left_ee_pose": left_ee_pose,
            "right_ee_pose": right_ee_pose,
            # object pose: dim=7
            "object_pose": object_pose,
            # goal pose: dim=7
            "goal_pose": torch.cat([self.goal_pos, self.goal_rot], dim=-1),
            "last_joints": joint_pos,
        }
        if self.cfg.enable_camera:
            # image observations: N*(H*W*C)
            obs["front_rgb"] = self._front_camera.data.output["rgb"].clone()[..., :3]
            obs["front_depth"] = (
                self._front_camera.data.output["distance_to_image_plane"]
                .clone()
                .unsqueeze(-1)
            )
            obs["left_rgb"] = self._left_wrist_camera.data.output["rgb"].clone()[
                ..., :3
            ]
            obs["left_depth"] = (
                self._left_wrist_camera.data.output["distance_to_image_plane"]
                .clone()
                .unsqueeze(-1)
            )
            obs["right_rgb"] = self._right_wrist_camera.data.output["rgb"].clone()[
                ..., :3
            ]
            obs["right_depth"] = (
                self._right_wrist_camera.data.output["distance_to_image_plane"]
                .clone()
                .unsqueeze(-1)
            )

        return {"policy": obs}

    def _process_joint_value(self):
        joint_pos = self._robot.data.joint_pos.clone()
        # print("[obs] raw joint_pos: ", joint_pos)
        l_arm_joint_pos = joint_pos[:, self.left_arm_joint_ids]
        r_arm_joint_pos = joint_pos[:, self.right_arm_joint_ids]
        l_gripper_joint_pos = joint_pos[:, self.left_gripper_joint_ids]
        r_gripper_joint_pos = joint_pos[:, self.right_gripper_joint_ids]

        self._translate_gripper_joints_to_state(
            l_gripper_joint_pos, r_gripper_joint_pos
        )
        joint_pos = torch.cat(
            [
                l_arm_joint_pos,
                self.left_gripper_state,
                r_arm_joint_pos,
                self.right_gripper_state,
            ],
            dim=-1,
        )
        # print("[obs] processed joint_pos: ", joint_pos)

        joint_vel = self._robot.data.joint_vel.clone()
        # print("[obs] raw joint_vel: ", joint_vel)
        l_arm_joint_vel = joint_vel[:, self.left_arm_joint_ids]
        r_arm_joint_vel = joint_vel[:, self.right_arm_joint_ids]
        l_gripper_joint_vel = joint_vel[:, self.left_gripper_joint_ids]
        r_gripper_joint_vel = joint_vel[:, self.right_gripper_joint_ids]
        # normalize gripper joint velocity
        l_gripper_joint_vel = l_gripper_joint_vel[:, 0] / (
            self.gripper_open - self.gripper_close
        )
        r_gripper_joint_vel = r_gripper_joint_vel[:, 0] / (
            self.gripper_open - self.gripper_close
        )

        joint_vel = torch.cat(
            [
                l_arm_joint_vel,
                l_gripper_joint_vel.view(-1, 1),
                r_arm_joint_vel,
                r_gripper_joint_vel.view(-1, 1),
            ],
            dim=-1,
        )
        # print("[obs] processed joint_vel: ", joint_vel)
        return joint_pos, joint_vel

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # object dropped, object is below the table
        # dropped = self._object.data.root_pos_w[:, 2] < 0.9
        # reached the goal pose
        reached = self._object_reached_goal()
        # terminated = torch.logical_or(dropped, reached)
        # time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return reached, time_out

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        reward = self._compute_reward()
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset robot
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.1, 0.1, (len(env_ids), self._robot.num_joints), device=self.device
        )
        joint_pos = torch.clamp(
            joint_pos, self.robot_joint_lower_limits, self.robot_joint_upper_limits
        )
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # reset object
        object_default_state = self._object.data.default_root_state[env_ids].clone()
        pose_range = {
            "x": (-0.05, 0.10),
            "y": (-0.1, 0.1),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.4, 0.4),
        }
        range_list = [
            pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        random_noise = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        object_default_state[:, 0:3] = (
            object_default_state[:, :3]
            + random_noise[:, :3]
            + self.scene.env_origins[env_ids]
        )
        orientation_noise = quat_from_euler_xyz(
            random_noise[:, 3], random_noise[:, 4], random_noise[:, 5]
        )
        object_default_state[:, 3:7] = quat_mul(
            object_default_state[:, 3:7], orientation_noise
        )

        self._object.write_root_state_to_sim(object_default_state, env_ids=env_ids)
        self.init_pos = object_default_state

        # print("reset object pose: ", object_default_state)

        # reset goal
        ranges[0] = torch.tensor([0.4, 0.4], device=self.device)  # x [0.35, 0.45]
        ranges[1] = torch.tensor([0.0, 0.0], device=self.device)  # y [-0.1, 0.1]
        ranges[2] = torch.tensor([1.25, 1.25], device=self.device)  # z [1.15, 1.35]
        ranges[5] = torch.tensor([0.0, 0.0], device=self.device)  # yaw
        random_delta = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        # should add env_origins for visualization
        self.goal_pos[env_ids] = random_delta[:, :3]
        self.goal_rot[env_ids] = quat_from_euler_xyz(
            random_delta[:, 3], random_delta[:, 4], random_delta[:, 5]
        )
        # print("reset goal pose: ", self.goal_pos, self.goal_rot)

        # reset success flag
        self.succ[env_ids] = 0

    # auxiliary methods
    def _process_action(self, actions: torch.Tensor):
        # left_arm: 6 joints
        # left_gripper: 2 joint
        # right_arm: 6 joints
        # right_gripper: 2 joints
        # gripper_state: open: 1, close: -1
        self.actions = actions.clone()
        # print("recieved actions: ", self.actions)

        if self.action_type == "joint_position":
            self.left_arm_joint_pos_target = self.actions[:, :6]
            self.right_arm_joint_pos_target = self.actions[:, 7:13]

            l_gripper_action = self.actions[:, 6].view(-1, 1)
            r_gripper_action = self.actions[:, 13].view(-1, 1)
            self._translate_gripper_state_to_joints(l_gripper_action, r_gripper_action)

        elif self.action_type == "ik_abs" or self.action_type == "ik_rel":
            if self.action_type == "ik_abs":
                l_arm_actions = self.actions[:, :7]
                l_gripper_action = self.actions[:, 7].view(-1, 1)
                r_arm_actions = self.actions[:, 8:15]
                r_gripper_action = self.actions[:, 15].view(-1, 1)
            elif self.action_type == "ik_rel":
                l_arm_actions = self.actions[:, :6]
                l_gripper_action = self.actions[:, 6].view(-1, 1)
                r_arm_actions = self.actions[:, 7:13]
                r_gripper_action = self.actions[:, 13].view(-1, 1)

            # mapping gripper state (1/-1, or True/False) to joint position
            self._translate_gripper_state_to_joints(l_gripper_action, r_gripper_action)

            # compute arm joint position using differential IK
            self._compute_arm_joints(l_arm_actions, r_arm_actions)

            # clamp the joint position targets
        else:
            raise ValueError(f"Unknown action type '{self.action_type}'")

    def _object_reached_goal(self):
        distance = self._distance_to_goal()
        reached = distance < self.cfg.goal_threshold
        # print("***** reached goal: {} *****".format(reached))

        return reached

    def _compute_reward(self):
        # reward_reaching_object_scale = 1.0
        # reward_lifted_object_scale = 15.0
        # reward_tracking_goal_coarse_scale = 16.0
        # reward_tracking_goal_fine_scale = 5.0
        # reward_penalize_action_rate = -1e-4
        # reward_penalize_joint_vel = -1e-4

        # 1. reward for reaching the object
        # get object position
        object_pos_w = self._object.data.root_pos_w
        object_quat_w = self._object.data.root_quat_w
        position_offset = torch.zeros_like(object_pos_w)
        position_offset[:, 1] = 0.15
        object_left_pos_w, _ = combine_frame_transforms(
            object_pos_w, object_quat_w, position_offset
        )
        position_offset[:, 1] = -0.15
        object_right_pos_w, _ = combine_frame_transforms(
            object_pos_w, object_quat_w, position_offset
        )
        # get end-effector position
        left_ee_pos_w = self._left_ee_frame.data.target_pos_w[..., 0, :]
        right_ee_pos_w = self._right_ee_frame.data.target_pos_w[..., 0, :]
        # compute the distance of the end-effector to the object
        object_left_ee_distance = torch.norm(object_left_pos_w - left_ee_pos_w, dim=1)
        object_right_ee_distance = torch.norm(
            object_right_pos_w - right_ee_pos_w, dim=1
        )
        object_ee_distance = object_left_ee_distance + object_right_ee_distance
        # compute reaching object reward
        reward_reaching_object = 1 - torch.tanh(object_ee_distance / 0.1)

        # 2. reward for lifting the object
        reward_lifted_object = torch.where(
            object_pos_w[:, 2] > self.cfg.minimal_height, 1.0, 0.0
        )

        # 3. reward for moving to the goal pose
        object_goal_distance = self._distance_to_goal()
        reward_tracking_goal_coarse = (object_pos_w[:, 2] > self.cfg.minimal_height) * (
            1 - torch.tanh(object_goal_distance / (self.cfg.goal_threshold * 2))
        )
        reward_tracking_goal_fine = (object_pos_w[:, 2] > self.cfg.minimal_height) * (
            1 - torch.tanh(object_goal_distance / self.cfg.goal_threshold)
        )

        # 4. penalize action rate
        reward_action_penalty = torch.sum(torch.square(self.actions), dim=1)
        # reward_penalize_action_rate = torch.sum(
        #     torch.square(self.actions - self.prev_actions), dim=1
        # )

        # 5. penalize joint velocity
        reward_joint_vel_penalty = torch.sum(
            torch.square(self._robot.data.joint_vel[:, :]), dim=1
        )

        rewards = (
            self.cfg.reward_reaching_object_scale * reward_reaching_object
            + self.cfg.reward_lifting_object_scale * reward_lifted_object
            + self.cfg.reward_tracking_goal_coarse_scale * reward_tracking_goal_coarse
            + self.cfg.reward_tracking_goal_fine_scale * reward_tracking_goal_fine
            + self.cfg.reward_action_penalty_scale * reward_action_penalty
            + self.cfg.reward_joint_vel_penalty_scale * reward_joint_vel_penalty
        )

        self.extras["log"] = {
            "reaching_object": (
                self.cfg.reward_reaching_object_scale * reward_reaching_object
            ).mean(),
            "lifting_object": (
                self.cfg.reward_lifting_object_scale * reward_lifted_object
            ).mean(),
            "tracking_goal_coarse": (
                self.cfg.reward_tracking_goal_coarse_scale * reward_tracking_goal_coarse
            ).mean(),
            "tracking_goal_fine": (
                self.cfg.reward_tracking_goal_fine_scale * reward_tracking_goal_fine
            ).mean(),
            "action_penalty": (
                self.cfg.reward_action_penalty_scale * reward_action_penalty
            ).mean(),
            "joint_vel_penalty": (
                self.cfg.reward_joint_vel_penalty_scale * reward_joint_vel_penalty
            ).mean(),
        }

        return rewards

    def _distance_to_goal(self):
        # get object position in the world frame
        object_pos_w = self._object.data.root_pos_w[:, :3]
        # compute goal position in the world frame
        goal_pos_w, _ = combine_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
            self.goal_pos,
        )
        # distance of the object to the goal pose
        distance = torch.norm(goal_pos_w - object_pos_w, dim=-1)
        return distance

    def _setup_ik_controller(self):
        if self.cfg.action_type == "ik_abs":
            # absolute mode, action_dim is 7: x, y, z, qw, qx, qy, qz
            use_relative_mode = False
        elif self.cfg.action_type == "ik_rel":
            # relative mode, action_dim is 6: dx, dy, dz, droll, dpitch, dyaw
            use_relative_mode = True
        else:
            print(
                "ActionType is {}, NO need to create IK controllers".format(
                    self.cfg.action_type
                )
            )
            return

        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=use_relative_mode, ik_method="dls"
        )
        self.l_diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=self.num_envs, device=self.device
        )
        self.r_diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=self.num_envs, device=self.device
        )

    def _compute_frame_pose(self):
        left_ee_pose_w = self._robot.data.body_state_w[:, self.left_ee_body_id, :7]
        right_ee_pose_w = self._robot.data.body_state_w[:, self.right_ee_body_id, :7]

        root_pose_w = self._robot.data.root_state_w[:, :7]

        left_ee_pos_b, left_ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            left_ee_pose_w[:, 0:3],
            left_ee_pose_w[:, 3:7],
        )
        right_ee_pos_b, right_ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            right_ee_pose_w[:, 0:3],
            right_ee_pose_w[:, 3:7],
        )

        left_ee_pos_b, left_ee_quat_b = combine_frame_transforms(
            left_ee_pos_b, left_ee_quat_b, self.ee_offset_pos, self.ee_offset_quat
        )
        right_ee_pos_b, right_ee_quat_b = combine_frame_transforms(
            right_ee_pos_b, right_ee_quat_b, self.ee_offset_pos, self.ee_offset_quat
        )

        return left_ee_pos_b, left_ee_quat_b, right_ee_pos_b, right_ee_quat_b

    def _compute_left_ee_jacobian(self):
        left_jacobian = self._robot.root_physx_view.get_jacobians()[
            :, self.left_ee_jacobi_idx, :, self.left_arm_joint_ids
        ]
        left_jacobian[:, 0:3, :] += torch.bmm(
            -skew_symmetric_matrix(self.ee_offset_pos), left_jacobian[:, 3:, :]
        )
        left_jacobian[:, 3:, :] = torch.bmm(
            matrix_from_quat(self.ee_offset_quat), left_jacobian[:, 3:, :]
        )
        return left_jacobian

    def _compute_right_ee_jacobian(self):
        right_jacobian = self._robot.root_physx_view.get_jacobians()[
            :, self.right_ee_jacobi_idx, :, self.right_arm_joint_ids
        ]
        right_jacobian[:, 0:3, :] += torch.bmm(
            -skew_symmetric_matrix(self.ee_offset_pos), right_jacobian[:, 3:, :]
        )
        right_jacobian[:, 3:, :] = torch.bmm(
            matrix_from_quat(self.ee_offset_quat), right_jacobian[:, 3:, :]
        )

        return right_jacobian

    def _compute_arm_joints(
        self, l_arm_actions: torch.Tensor, r_arm_actions: torch.Tensor
    ):
        # process arm action
        l_ee_pose_curr, l_ee_quat_curr, r_ee_pose_curr, r_ee_quat_curr = (
            self._compute_frame_pose()
        )
        self.l_diff_ik_controller.set_command(
            l_arm_actions, l_ee_pose_curr, l_ee_quat_curr
        )
        self.r_diff_ik_controller.set_command(
            r_arm_actions, r_ee_pose_curr, r_ee_quat_curr
        )
        l_joint_pos_curr = self._robot.data.joint_pos[:, self.left_arm_joint_ids]
        r_joint_pos_curr = self._robot.data.joint_pos[:, self.right_arm_joint_ids]

        if l_ee_quat_curr.norm() != 0:
            l_jacobian = self._compute_left_ee_jacobian()
            self.left_arm_joint_pos_target = self.l_diff_ik_controller.compute(
                l_ee_pose_curr, l_ee_quat_curr, l_jacobian, l_joint_pos_curr
            )
        else:
            print("use current joint position")
            self.left_arm_joint_pos_target = l_joint_pos_curr.clone()

        if r_ee_quat_curr.norm() != 0:
            r_jacobian = self._compute_right_ee_jacobian()
            self.right_arm_joint_pos_target = self.r_diff_ik_controller.compute(
                r_ee_pose_curr, r_ee_quat_curr, r_jacobian, r_joint_pos_curr
            )
        else:
            print("use current joint position")
            self.right_arm_joint_pos_target = r_joint_pos_curr.clone()


    # @torch.jit.script
    def _translate_gripper_state_to_joints(
        self, l_gripper_action: torch.Tensor, r_gripper_action: torch.Tensor
    ):
        # if gripper action is False or -1, close the gripper
        if l_gripper_action.dtype == torch.bool:
            l_binary_mask = l_gripper_action == 0
        else:
            l_binary_mask = l_gripper_action <= (self.gripper_open - 0.005)
        self.left_gripper_joint_pos_target = torch.where(
            l_binary_mask,
            self.gripper_close_joints,
            self.gripper_open_joints,
        )

        if r_gripper_action.dtype == torch.bool:
            r_binary_mask = r_gripper_action == 0
        else:
            r_binary_mask = r_gripper_action <= (self.gripper_open - 0.005)
        self.right_gripper_joint_pos_target = torch.where(
            r_binary_mask, self.gripper_close_joints, self.gripper_open_joints
        )

        

    def _translate_gripper_joints_to_state(
        self, l_gripper_joints: torch.Tensor, r_gripper_joints: torch.Tensor
    ):
        l_binary_mask = l_gripper_joints[:, 0] <= (self.gripper_open - 0.005)
        r_binary_mask = r_gripper_joints[:, 0] <= (self.gripper_open - 0.005)
        l_binary_mask = l_binary_mask.view(-1, 1)
        r_binary_mask = r_binary_mask.view(-1, 1)

        self.left_gripper_state = torch.where(
            l_binary_mask, self.gripper_close_state, self.gripper_open_state
        )
        self.right_gripper_state = torch.where(
            r_binary_mask, self.gripper_close_state, self.gripper_open_state
        )


    def _setup_robot(self):
        # create a buffer for joint position actions
        self.robot_joint_pos_target = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device
        )

        # create left/right arm scene entity cfg
        self.left_arm_entity_cfg = SceneEntityCfg(
            "robot", joint_names=["left_arm_.*"], body_names=["left_arm_link6"]
        )
        self.right_arm_entity_cfg = SceneEntityCfg(
            "robot", joint_names=["right_arm_.*"], body_names=["right_arm_link6"]
        )
        self.left_arm_entity_cfg.resolve(self.scene)
        self.right_arm_entity_cfg.resolve(self.scene)

        # get left/right end-effector body id
        self.left_ee_body_id = self.left_arm_entity_cfg.body_ids[0]
        self.right_ee_body_id = self.right_arm_entity_cfg.body_ids[0]
        if self._robot.is_fixed_base:
            self.left_ee_jacobi_idx = self.left_ee_body_id - 1
            self.right_ee_jacobi_idx = self.right_ee_body_id - 1
        else:
            self.left_ee_jacobi_idx = self.left_ee_body_id
            self.right_ee_jacobi_idx = self.right_ee_body_id
        # get left/right arm joint ids
        self.left_arm_joint_ids = self.left_arm_entity_cfg.joint_ids
        self.right_arm_joint_ids = self.right_arm_entity_cfg.joint_ids
        self.num_arm_joints = len(self.left_arm_joint_ids)
        if self.num_arm_joints != len(self.right_arm_joint_ids):
            raise ValueError(
                "The number of left and right arm joints should be the same."
            )

        # create a buffer for left/right arm joint position
        self.left_arm_joint_pos_target = torch.zeros(
            (self.num_envs, self.num_arm_joints),
            dtype=torch.float,
            device=self.device,
        )
        self.right_arm_joint_pos_target = torch.zeros(
            (self.num_envs, self.num_arm_joints),
            dtype=torch.float,
            device=self.device,
        )

        # build left/right gripper entity cfg
        self.left_gripper_entity_cfg = SceneEntityCfg(
            "robot", joint_names=["left_gripper_.*"]
        )
        self.right_gripper_entity_cfg = SceneEntityCfg(
            "robot", joint_names=["right_gripper_.*"]
        )
        self.left_gripper_entity_cfg.resolve(self.scene)
        self.right_gripper_entity_cfg.resolve(self.scene)

        # get left/right gripper joint ids
        self.left_gripper_joint_ids = self.left_gripper_entity_cfg.joint_ids
        self.right_gripper_joint_ids = self.right_gripper_entity_cfg.joint_ids
        self.num_gripper_joints = len(self.left_gripper_joint_ids)
        if self.num_gripper_joints != len(self.right_gripper_joint_ids):
            raise ValueError(
                "The number of left and right gripper joints should be the same."
            )

        self.gripper_open = 0.03
        self.gripper_close = 0.0
        self.gripper_open_joints = torch.full(
            (self.num_gripper_joints,), self.gripper_open, device=self.device
        )
        self.gripper_close_joints = torch.full(
            (self.num_gripper_joints,), self.gripper_close, device=self.device
        )
        self.gripper_open_state = torch.ones((1,), device=self.device)
        self.gripper_close_state = torch.zeros((1,), device=self.device)

        # create gripper joints buffer, default open (1 for open, -1/0 for close)
        self.left_gripper_joint_pos_target = torch.zeros(
            (self.num_envs, self.num_gripper_joints),
            device=self.device,
        )
        self.right_gripper_joint_pos_target = torch.zeros(
            (self.num_envs, self.num_gripper_joints),
            device=self.device,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "vis_markers"):
                self.vis_markers = VisualizationMarkers(self.cfg.vis_markers_cfg)
            self.vis_markers.set_visibility(True)
        else:
            if hasattr(self, "vis_markers"):
                self.vis_markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        # visualize the goal pose
        self.goal_pos_vis = self.goal_pos + self.scene.env_origins
        self.obj_pos_vis = self._object.data.root_pos_w
        marker_translations = torch.cat([self.goal_pos_vis, self.obj_pos_vis], dim=0)
        marker_orientations = torch.cat(
            [self.goal_rot, self._object.data.root_quat_w], dim=0
        )

        self.vis_markers.visualize(
            translations=marker_translations,
            orientations=marker_orientations,
            marker_indices=self.marker_indices,
        )

from __future__ import annotations

import torch
import copy
import random
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
import math
from .lift_env_cfg import (
    R1LiftCubeAbsEnvCfg,
    R1LiftCubeRelEnvCfg,
    R1LiftBinAbsEnvCfg,
    R1LiftBinRelEnvCfg,
    R1MultiFruitAbsEnvCfg,
)

class R1MultiFruitEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_observations()
    #   |-- _get_rewards()
    #   |-- _get_dones()
    #   |-- _reset_idx(env_ids)

    cfg: (
        R1LiftCubeAbsEnvCfg
        | R1LiftCubeRelEnvCfg
        | R1LiftBinAbsEnvCfg
        | R1LiftBinRelEnvCfg
        | R1MultiFruitAbsEnvCfg
    )
    """ Lift an object to a target position.
    """

    def __init__(
        self,
        cfg: (
            R1LiftCubeAbsEnvCfg
            | R1LiftCubeRelEnvCfg
            | R1LiftBinAbsEnvCfg
            | R1LiftBinRelEnvCfg
            | R1MultiFruitAbsEnvCfg
        ),
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

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

        self.object_id = random.randint(0, 2)
        self.init_pos = torch.zeros(size=(self.num_envs, 3), device=self.device)

        print("R1LiftEnv is initialized. ActionType: ", self.action_type)

    def _setup_scene(self):
        self._object = [0]*4 
        self._robot = Articulation(self.cfg.robot_cfg)
        self._drop_height = 0.96
        # add robot, object

        object1_cfg = copy.deepcopy(self.cfg.carrot_cfg)
        object1_pos = (0.35, -0.35, self._drop_height)
        object1_cfg.init_state.pos = object1_pos
        object1_cfg.spawn.scale = (0.3, 0.3, 0.3)
        self._object[0] = RigidObject(object1_cfg)
        self._object.append(RigidObject(object1_cfg))

        object2_cfg = copy.deepcopy(self.cfg.banana_cfg)
        object2_pos = (0.6, -0.6, self._drop_height)
        object2_cfg.init_state.pos = object2_pos
        object2_cfg.spawn.scale = (0.3, 0.3, 0.3)
        self._object[1] = RigidObject(object2_cfg)
        self._object.append(RigidObject(object2_cfg))

        object3_cfg = copy.deepcopy(object2_cfg)
        new_pos = (0.6, 0.6, self._drop_height)
        object3_cfg.init_state.pos = new_pos
        prim_path = "/World/envs/env_.*/banana3"
        object3_cfg.prim_path = prim_path
        self._object[2] = RigidObject(object3_cfg)

        basket_cfg = copy.deepcopy(self.cfg.basket_cfg)
        new_pos = (0.45, 0.05, 1.05)
        basket_cfg.spawn.scale = (0.4, 0.4, 0.4)
        basket_cfg.init_state.pos = new_pos
        basket_cfg.init_state.rot = (0.707, 0.0, 0.0, 0.707)
        prim_path = "/World/envs/env_.*/basket"
        basket_cfg.prim_path = prim_path
        self._object[3] = RigidObject(basket_cfg)

        # add table which is a static object
        if self.cfg.table_cfg.spawn is not None:
            self.cfg.table_cfg.spawn.scale = (0.09, 0.09, 0.09)
            self.cfg.table_cfg.spawn.func(
                self.cfg.table_cfg.prim_path,
                self.cfg.table_cfg.spawn,
                translation=self.cfg.table_cfg.init_state.pos,
                orientation=self.cfg.table_cfg.init_state.rot,
            )

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
        self.scene.rigid_objects["object0"] = self._object[0]
        self.scene.rigid_objects["object1"] = self._object[1]
        self.scene.rigid_objects["object2"] = self._object[2]
        self.scene.rigid_objects["object3"] = self._object[3]
        self.scene.sensors["left_ee_frame"] = self._left_ee_frame
        self.scene.sensors["right_ee_frame"] = self._right_ee_frame
        self.scene.extras["table"] = XFormPrimView(
            self.cfg.table_cfg.prim_path, reset_xform_properties=False
        )

        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(color=(1.0, 1.0, 1.0))
        )

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
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

        joint_pos, joint_vel = self._process_joint_value()

        # get object pose
        object_pos = self._object[self.object_id].data.root_pos_w - self.scene.env_origins
        object_pose = torch.cat([object_pos, self._object[self.object_id].data.root_quat_w], dim=-1)

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

        joint_vel = self._robot.data.joint_vel.clone()
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
        return joint_pos, joint_vel

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        reached = self._object_reached_goal()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        right_ee_z_pos = self._get_right_ee_pose()[:, 2]
        reached_total = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out_total = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for ith in range(self.num_envs):
            reached_total[ith] = reached[ith] and right_ee_z_pos[ith] > 1.20
            time_out_total[ith] = time_out[ith] or self.succ[ith]
        return reached_total, time_out_total

    def _get_right_ee_pose(self):
        right_ee_pos = (
            self._right_ee_frame.data.target_pos_w[..., 0, :] - self.scene.env_origins
        )
        right_ee_pose = torch.cat(
            [right_ee_pos, self._right_ee_frame.data.target_quat_w[..., 0, :]], dim=-1
        )
        return right_ee_pose
    def _get_rewards(self) -> torch.Tensor:
        reward = self._compute_reward()
        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.object_id = random.randint(0, 0)
        print(f"object_id:{self.object_id}")

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

        object_default_state = self._object[0].data.default_root_state[env_ids].clone()
        env_num = object_default_state.shape[0]  

        initial_x = object_default_state[:, 0] 
        initial_y = object_default_state[:, 1] 

        #arbitarily set the offsets among 3x3 discrete values
        x_offsets = torch.tensor([0.0, 0.05, 0.1], device=self.device)
        y_offsets = torch.tensor([0.0, 0.05, 0.1], device=self.device)
        
        random_x_indices = torch.randint(0, len(x_offsets), (env_num,), device=self.device)
        random_y_indices = torch.randint(0, len(y_offsets), (env_num,), device=self.device)
        
        selected_x_offsets = x_offsets[random_x_indices]  # shape=[env_num]
        selected_y_offsets = y_offsets[random_y_indices]  # shape=[env_num]
        
        object_default_state[:, 0] = initial_x + selected_x_offsets
        object_default_state[:, 1] = initial_y + selected_y_offsets
        object_default_state[:, 0:3] = object_default_state[:, 0:3] + self.scene.env_origins[env_ids]

        #arbitarily set the yaw among 5 discrete values (0, 45, 90, 135, 180) 
        yaw_options = torch.tensor([0, math.radians(45), math.radians(90), math.radians(135), math.radians(180)], device=self.device)
        
        random_yaw = random.choices(yaw_options, k=env_num)  
        random_yaw = torch.tensor(random_yaw, device=self.device)  
        
        rolls = torch.zeros(env_num, device=self.device)  
        pitches = torch.zeros(env_num, device=self.device)  

        #update the orientation of the object
        orientation_noise = quat_from_euler_xyz(
            rolls, pitches, random_yaw
        )
        object_default_state[:, 3:7] = quat_mul(
            object_default_state[:, 3:7], orientation_noise
        )
        self.init_pos[env_ids] = torch.cat((object_default_state[:, 0:2]- self.scene.env_origins[env_ids, 0:2], random_yaw.unsqueeze(1)), dim=1)


        self._object[0].write_root_state_to_sim(object_default_state, env_ids=env_ids)

        self.goal_pos[env_ids] = object_default_state[:, 0:3].clone() - self.scene.env_origins[env_ids]
        self.goal_pos[env_ids, 2] = 1.3
        self.goal_rot[env_ids] = object_default_state[:, 3:7].clone()

        object_default_state3 = self._object[3].data.default_root_state[env_ids].clone()
        object_default_state3[:, 0:3] = object_default_state3[:, 0:3] + self.scene.env_origins[env_ids]
        self._object[3].write_root_state_to_sim(object_default_state3, env_ids=env_ids)

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
        object_curr_pos = self._object[self.object_id].data.root_pos_w[:, :3]
        basket_pos = self._object[3].data.root_pos_w[:, :3]
        reached = self._within_basket(object_curr_pos, basket_pos)
        return reached

    def _within_basket(self, obj_pos, basket_pos):
        result = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        ith = 0
        for obj_pos_ith, basket_pos_ith in zip(obj_pos, basket_pos):
            result_ith = ((obj_pos_ith[0] > basket_pos_ith[0] - 0.1) and (obj_pos_ith[0] < basket_pos_ith[0] + 0.1) and 
                        (obj_pos_ith[1] > basket_pos_ith[1] - 0.15) and (obj_pos_ith[1] < basket_pos_ith[1] + 0.15)  and 
                        (obj_pos_ith[2] > basket_pos_ith[2] - 0.1) and (obj_pos_ith[2] < basket_pos_ith[2] + 0.05))
            result[ith] = result_ith
            ith += 1
        return result



    def _compute_reward(self):
        return torch.zeros(self.num_envs, device=self.device)
       
    def _distance_to_goal(self):
        object_pos_w = self._object[self.object_id].data.root_pos_w[:, :3]
        goal_pos_w = self._object[3].data.root_pos_w[:, :3]
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
        self.obj_pos_vis = self._object[self.object_id].data.root_pos_w
        marker_translations = torch.cat([self.goal_pos_vis, self.obj_pos_vis], dim=0)
        marker_orientations = torch.cat(
            [self.goal_rot, self._object[self.object_id].data.root_quat_w], dim=0
        )

        self.vis_markers.visualize(
            translations=marker_translations,
            orientations=marker_orientations,
            marker_indices=self.marker_indices,
        )

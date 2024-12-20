from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """_summary_

    Args:
        env (ManagerBasedRLEnv): _description_
        minimal_height (float): _description_
        object_cfg (SceneEntityCfg, optional): _description_. Defaults to SceneEntityCfg("object").

    Returns:
        torch.Tensor: _description_
    """
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    right_ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """_summary_

    Args:
        env (ManagerBasedRLEnv): _description_
        std (float): _description_
        object_cfg (SceneEntityCfg, optional): _description_. Defaults to SceneEntityCfg("object").
        left_ee_frame_cfg (SceneEntityCfg, optional): _description_. Defaults to SceneEntityCfg("ee_frame").
        right_ee_frame_cfg (SceneEntityCfg, optional): _description_. Defaults to SceneEntityCfg("ee_frame").

    Returns:
        torch.Tensor: _description_
    """
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    left_ee_frame: FrameTransformer = env.scene[left_ee_frame_cfg.name]
    right_ee_frame: FrameTransformer = env.scene[right_ee_frame_cfg.name]

    # Target object position: (num_envs, 3)
    box_pos_w = object.data.root_pos_w
    box_left_handle_pos_w = box_pos_w - 0.2
    box_right_handle_pos_w = box_pos_w + 0.2

    # End-effector position: (num_envs, 3)
    left_ee_w = left_ee_frame.data.target_pos_w[..., 0, :]
    right_ee_w = right_ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_left_ee_distance = torch.norm(box_left_handle_pos_w - left_ee_w, dim=1)
    object_right_ee_distance = torch.norm(box_right_handle_pos_w - right_ee_w, dim=1)
    object_ee_distance = object_left_ee_distance + object_right_ee_distance

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

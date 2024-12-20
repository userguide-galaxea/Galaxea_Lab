from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env (ManagerBasedRLEnv): _description_
        command_name (str, optional): _description_. Defaults to "object_pose".
        threshold (float, optional): _description_. Defaults to 0.02.
        robot_cfg (SceneEntityCfg, optional): _description_. Defaults to SceneEntityCfg("robot").
        object_cfg (SceneEntityCfg, optional): _description_. Defaults to SceneEntityCfg("object").

    Returns:
        torch.Tensor: _description_
    """

    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b
    )
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    return distance < threshold

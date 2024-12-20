# -*- coding: utf-8 -*-
# Copyright (c) 2024 Galaxea

"""Configuration for the  loong robot (production date: 0604)
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR


##
# Configuration
##

LOONG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Loong/loong.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "l_hip_roll": 0.0,
            "l_hip_pitch": 0.0,
            "l_hip_yaw": 0.0,
            "l_knee_pitch": 0.0,
            "l_ankle_pitch": 0.0,
            "l_ankle_roll": 0.0,

            "r_hip_roll": 0.0,
            "r_hip_pitch": 0.0,
            "r_hip_yaw": 0.0,
            "r_knee_pitch": 0.0,
            "r_ankle_pitch": 0.0,
            "r_ankle_roll": 0.0,
        },
        pos=(0.0, 0.0, 1.04),
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                ".*_hip.*": 20.0,
                ".*_ankle.*": 10.0,
                ".*_knee.*": 10.0,
            },
            damping={
                ".*_hip.*": 5.0,
                ".*_ankle.*": 5.0,
                ".*_knee.*": 5.0,
            },
        ),

    },
    soft_joint_pos_limit_factor=1.0,
)

LOONG_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Camera",  # should be replaced with the actual parent frame
    update_period=1 / 60.0,  # 30 Hz
    height=240,
    width=320,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=12,
        focus_distance=100.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 100),
    ),
    offset=CameraCfg.OffsetCfg(  # offset from the parent frame
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="ros",
    ),
)

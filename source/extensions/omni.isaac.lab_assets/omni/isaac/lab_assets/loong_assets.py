# -*- coding: utf-8 -*-
# Copyright (c) 2024 Galaxea

"""Configuration for the Galaxea R1 robot (production date: 0604)
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR
from omni.isaac.lab.assets import (
    RigidObjectCfg,
    AssetBaseCfg,
)
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg


SHELF_CFG_LOONG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Shelf",
    spawn=UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/loong_assets/shelf.usd",
        scale=(1, 1, 1),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
            rigid_body_enabled=True,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0, 1.5, 0.98),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)




BOTTLE_BOX_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/BottleBox",
    spawn=UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/loong_assets/box3.usd",
        scale=(1, 1, 1),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
            rigid_body_enabled=True,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.3, 1.5, 1),
        rot=(0.0, 0, 1, 0),
    ),
)

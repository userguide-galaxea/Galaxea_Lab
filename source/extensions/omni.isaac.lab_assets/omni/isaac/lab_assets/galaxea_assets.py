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


##
# Configuration
##

# cubes
DEX_CUBE_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/block/DexCube/dex_cube_instanceable.usd",
        scale=(0.6, 6.0, 0.6),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.36, 0.0, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

MULTI_COLOR_CUBE_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/block/MultiColorCube/multi_color_cube_instanceable.usd",
        scale=(0.6, 6.0, 0.6),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.36, 0.0, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

T_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/block/T/t_block.usd",
        scale=(0.6, 6.0, 0.6),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.36, 0.0, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

# bin
KLT_BIN_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/KLT_Bin/small_KLT.usd",
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 1.05),
        rot=(1.0, 0, 0, 0),
    ),
)

BASKET_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Basket",
    debug_vis=True,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/fruits/basket.usd",
        scale=(0.3, 0.3, 0.3),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.46, 0.4, 1.0),
        rot=(1.0, 0, 0, 0.),
    ),
)

# tables
BIGYM_TABLE_CFG: AssetBaseCfg = AssetBaseCfg(
    prim_path="/World/envs/env_.*/Table",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/table/table.usd",
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0.5, 0.0, 0.0),
        rot=(-0.70711, 0.0, 0.0, 0.70711),
    ),
)

OAK_TABLE_CFG: AssetBaseCfg = AssetBaseCfg(
    prim_path="/World/envs/env_.*/Table",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/table/OakTableLarge.usd",
        scale=(0.005, 0.008, 0.009),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0.55, 0.0, 0.0),
        rot=(-0.70711, 0.0, 0.0, 0.70711),
    ),
)

LOW_FULL_DESK_CFG: AssetBaseCfg = AssetBaseCfg(
    prim_path="/World/envs/env_.*/Table",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/table/LowFullDesk.usd",
        scale=(0.01, 0.01, 0.0135),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0.52, 0.0, 0.0),
        rot=(-0.70711, 0.0, 0.0, 0.70711),
    ),
)

CARROT_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/carrot",
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"/home/user/zhr_workspace/IsaacsimAsset/objects/fruits/fruit_v2/banana.usd",
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/fruits/carrot.usd",
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
            rigid_body_enabled=True
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.36, 0.0, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

BANANA_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/banana",
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"/home/user/zhr_workspace/IsaacsimAsset/objects/fruits/fruit_v2/banana.usd",
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/fruits/banana.usd",
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
            rigid_body_enabled=True
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.36, -0.5, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)



TABLE_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Table",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.52, 0.0, 0.0),
        rot=(-0.70711, 0.0, 0.0, 0.70711),
    ),
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/table/OakTableLarge.usd",
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/table/LowFullDesk.usd",
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/table/desk.usd",
        scale=(0.01, 0.01, 0.0135),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,  # 禁用重力，确保桌子不会因重力移动
        kinematic_enabled=True,
        retain_accelerations=False,  # 防止物体在碰撞后受到加速度影响
    ),
    ),
)

SEKTION_CABINET_CFG: AssetBaseCfg = AssetBaseCfg(
    prim_path="/World/envs/env_.*/Table",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
        scale=(1.0, 2.0, 1.2),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0.55, 0.0, 0.5),
        rot=(0.0, 0.0, 0.0, 1.0),  # for sektion cabinet
    ),
)

T_BLOCK_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/T",
    spawn=UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/block/T/t_block.usd",
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.6, 0.3, 1.2),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

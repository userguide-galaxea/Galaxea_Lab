# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import RigidObjectCfg, AssetBaseCfg
from omni.isaac.lab.sensors import FrameTransformerCfg, CameraCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils
from .utils import get_camera_rot_offset

from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_tasks.galaxea.manager_based.lift import mdp
from omni.isaac.lab_tasks.galaxea.manager_based.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets import GALAXEA_R1_HIGH_PD_CFG, GALAXEA_R1_HIGH_PD_GRIPPER_CFG
from omni.isaac.lab_assets import (
    GALAXEA_R1_HIGH_PD_CFG,
    GALAXEA_R1_HIGH_PD_GRIPPER_CFG,
    GALAXEA_CAMERA_CFG,
    DEX_CUBE_CFG,
    KLT_BIN_CFG,
    MULTI_COLOR_CUBE_CFG,
    T_CFG,
    BIGYM_TABLE_CFG,
    OAK_TABLE_CFG,
    LOW_FULL_DESK_CFG,
    SEKTION_CABINET_CFG,
    TABLE_CFG,
    # FRUIT_CFG,
    BASKET_CFG
)

@configclass
class R1LiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type (franka)
        self.actions.left_arm_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint.*"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.right_arm_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint.*"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.left_finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_gripper_.*"],
            open_command_expr={"left_gripper.*": 0.03},
            close_command_expr={"left_gripper_.*": 0.0},
        )
        self.actions.right_finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_gripper_axis.*"],
            open_command_expr={"right_gripper.*": 0.03},
            close_command_expr={"right_gripper_.*": 0.0},
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = "base_link"

        # Set the camera, 40 degree offset to see the table
        # rot_offset = get_camera_rot_offset(0.698132)
        rot_offset = get_camera_rot_offset(roll=0.7854)
        print("************front rot offset: ", rot_offset)
        self.scene.front_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link4/front_camera",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12,
                focus_distance=100.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.066, 0.0, 0.482),
                rot=rot_offset,
                convention="ros",
            ),
        )

        rot_offset = get_camera_rot_offset(roll=-0.5236, pitch=-3.14159)
        print("************left rot offset: ", rot_offset)
        self.scene.left_wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_arm_link6/left_wrist_camera",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10,
                focus_distance=100.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.068, 0.0, 0.057),
                rot=rot_offset,
                convention="ros",
            ),
        )

        rot_offset = get_camera_rot_offset(roll=-0.5236, pitch=-3.14159)
        print("************right rot offset: ", rot_offset)
        self.scene.right_wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_arm_link6/right_wrist_camera",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10,
                focus_distance=100.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.068, 0.0, 0.057),
                rot=rot_offset,
                convention="ros",
            ),
        )

        # # Listens to the required transforms
        left_marker_cfg = FRAME_MARKER_CFG.copy()
        left_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        left_marker_cfg.prim_path = "/Visuals/FrameTransformer/Left"
        self.scene.left_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_arm_link1",
            debug_vis=True,
            visualizer_cfg=left_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_arm_link6",
                    name="left_end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.15),
                    ),
                ),
            ],
        )

        right_marker_cfg = FRAME_MARKER_CFG.copy()
        right_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        right_marker_cfg.prim_path = "/Visuals/FrameTransformer/Right"
        self.scene.right_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_arm_link1",
            debug_vis=True,
            visualizer_cfg=right_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_arm_link6",
                    name="right_end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.15),
                    ),
                ),
            ],
        )


@configclass
class R1LiftCubeEnvCfg(R1LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = GALAXEA_R1_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # self.scene.table = AssetBaseCfg(
        #     prim_path="{ENV_REGEX_NS}/Table",
        #     spawn=sim_utils.UsdFileCfg(
        #         # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/table/table.usd",
        #         usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/table/table.usd",
                
        #     ),
        #     init_state=AssetBaseCfg.InitialStateCfg(
        #         pos=(0.5, 0.0, 0.0),
        #         rot=(-0.70711, 0.0, 0.0, 0.70711),  # for bigym table
        #     ),
        # )
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
                scale=(1.0, 2.0, 1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.55, 0.0, 0.5),
                rot=(0.0, 0.0, 0.0, 1.0),  # for sektion cabinet
            ),
        )
        # self.scene.table.spawn.func(
        #     "/World/table",
        #     self.scene.table.spawn,
        #     translation = self.scene.table.init_state.pos,
        #     orientation = self.scene.table.init_state.rot,
        # )

        # self.scene.extras["table"] = XFormPrimView(
        #     self.cfg.table_cfg.prim_path, reset_xform_properties=False
        # )


        # Set cube as object
        self.scene.object = RigidObjectCfg( 
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.36, 0.0, 1.0),
                rot=(1.0, 0, 0, 0),
            ),
            debug_vis=True,
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/block/DexCube/dex_cube_instanceable.usd",
                # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/block/MultiColorCube/multi_color_cube_instanceable.usd",
                scale=(0.6, 6.0, 0.6),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        self.scene.object2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.8, 0.0, 1.5),
                rot=(1.0, 0, 0, 0),
            ),
            debug_vis=True,
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/block/DexCube/dex_cube_instanceable.usd",
                # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/block/MultiColorCube/multi_color_cube_instanceable.usd",
                scale=(0.6, 6.0, 0.6),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )



        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=RigidObjectCfg.InitialStateCfg(
        #         pos=(0.4, 0.0, 1.0),
        #         rot=(1.0, 0.0, 0.0, 0.0),
        #     ),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/block/T/t_block.usd",
        #         scale=(0.8, 0.8, 0.8),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )


@configclass
class R1LiftCubeEnvCfg_PLAY(R1LiftCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
    


@configclass
class R1LiftBinEnvCfg(R1LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = GALAXEA_R1_HIGH_PD_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
                scale=(1.0, 2.0, 1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.55, 0.0, 0.5),
                rot=(0.0, 0.0, 0.0, 1.0),  # for sektion cabinet
            ),
        )

        # Set bin as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.4, 0.0, 1.0),
                rot=(1.0, 0, 0, 0),
            ),
            debug_vis=True,
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/KLT_Bin/small_KLT.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

@configclass
class R1LiftFruitEnvCfg(R1LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # breakpoint()
        self.scene.robot = GALAXEA_R1_HIGH_PD_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )


@configclass
class R1LiftBinEnvCfg_PLAY(R1LiftBinEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

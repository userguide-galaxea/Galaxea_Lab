from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg

# from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
)
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import (
    GroundPlaneCfg,
    UsdFileCfg,
)
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR

from . import mdp

###
# Scene definition
###


@configclass
class PushSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot with dual-arm, a table and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector: will be populated by agent env cfg
    left_ee_frame: FrameTransformerCfg = MISSING
    right_ee_frame: FrameTransformerCfg = MISSING

    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING
    # self.scene.object = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.4, 0.0, 1.0),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #     ),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/block/t_shape.usd",
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

    # Table
    table: AssetBaseCfg = MISSING

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


###
# MDP settings
###
@configclass
class CommandsCfg:
    """command terms for randomizing the object target pose."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.45),
            pos_y=(-0.1, 0.1),
            pos_z=(1.1, 1.25),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """action specifications for the MDP."""

    # left arm
    left_arm_joint_pos: mdp.JointPositionActionCfg = MISSING
    left_finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING
    # right arm
    right_arm_joint_pos: mdp.JointPositionActionCfg = MISSING
    right_finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group"""

        # joint_pos_raw = ObsTerm(func=mdp.joint_pos)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_pose = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events"""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),
                # "roll": (0.0, 0.0),
                # "pitch": (0.0, 0.0),
                "yaw": (-0.5, 0.5),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
            # "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    reaching_object = RewTerm(
        func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0
    )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.1, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


@configclass
class PushEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment

    Args:
        ManagerBasedRLEnvCfg (_type_): _description_
    """

    # scene settings
    scene: PushSceneCfg = PushSceneCfg(num_envs=8, env_spacing=2.5)

    # basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationCfg = TerminationCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 2
        self.episode_length_s = 3.7
        self.viewer.eye = (5.0, 5.0, 5.0)
        self.viewer.lookat = (0.8, 0.0, 1.5)

        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="create scene using the interactive scene interface"
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="number of environments to spawn"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import omni.isaac.lab.sim as sim_utils

from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
import copy

# from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR
from omni.isaac.lab_assets import (

    LOONG_CFG,

    BOTTLE_BOX_CFG,
    SHELF_CFG_LOONG,


)


@configclass
class LoongSceneCfg(InteractiveSceneCfg):
    """Configuration for Loong robot scene."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75)),
    )


    loong: ArticulationCfg = LOONG_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    shelf: RigidObjectCfg = SHELF_CFG_LOONG.copy()

    bottle_box: RigidObjectCfg = BOTTLE_BOX_CFG.copy()

    bottle_box2: RigidObjectCfg = copy.deepcopy(BOTTLE_BOX_CFG)
    bottle_box2.init_state.pos = (0.2, 1.8, 1.55)
    bottle_box2.init_state.rot = (1.0, 0, 0, 0)
    bottle_box2.prim_path = "/World/envs/env_.*/BottleBox2"






    # t_block: RigidObjectCfg = T_BLOCK_CFG.copy()


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["loong"]
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            joint_pos, joint_vel = (
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.01
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()

            print("[INFO]: Resetting robot state...")
        joint_pos_default = robot.data.default_joint_pos.clone()
        robot.set_joint_position_target(joint_pos_default)
        robot.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    scene_cfg = LoongSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()

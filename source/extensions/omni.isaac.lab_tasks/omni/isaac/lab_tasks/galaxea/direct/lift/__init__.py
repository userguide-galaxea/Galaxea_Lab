# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import os
import gymnasium as gym

from . import agents
from .lift_env import R1LiftEnv
from .pick_fruit_env import R1MultiFruitEnv
from .lift_env_cfg import (
    R1LiftCubeEnvCfg,
    R1LiftCubeAbsEnvCfg,
    R1LiftCubeRelEnvCfg,
    # R1LiftBinEnvCfg,
    R1LiftBinAbsEnvCfg,
    R1LiftBinRelEnvCfg,
    R1MultiFruitAbsEnvCfg,
)

##
# Register Gym environments.
##

gym.register(
    id="Isaac-R1-Multi-Fruit-IK-Abs-Direct-v0",
    entry_point="omni.isaac.lab_tasks.galaxea.direct.lift:R1MultiFruitEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": R1MultiFruitAbsEnvCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FrankaCabinetPPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-R1-Multi-Cube-IK-Abs-Direct-v0",
#     entry_point="omni.isaac.lab_tasks.galaxea.direct.lift:R1LiftEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": R1MultiLiftCubeAbsEnvCfg,
#         # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FrankaCabinetPPORunnerCfg,
#         # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-R1-Lift-Cube-Direct-v0",
#     entry_point="omni.isaac.lab_tasks.galaxea.direct.lift:R1LiftEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": R1LiftCubeEnvCfg,
#         # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FrankaCabinetPPORunnerCfg,
#         # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#     },
# )

gym.register(
    id="Isaac-R1-Lift-Cube-IK-Abs-Direct-v0",
    entry_point="omni.isaac.lab_tasks.galaxea.direct.lift:R1LiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": R1LiftCubeAbsEnvCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FrankaCabinetPPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-R1-Lift-Cube-IK-Rel-Direct-v0",
    entry_point="omni.isaac.lab_tasks.galaxea.direct.lift:R1LiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": R1LiftCubeRelEnvCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FrankaCabinetPPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-R1-Lift-Bin-Direct-v0",
#     entry_point="omni.isaac.lab_tasks.galaxea.direct.lift:R1LiftEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": R1LiftBinEnvCfg,
#         # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FrankaCabinetPPORunnerCfg,
#         # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#     },
# )

gym.register(
    id="Isaac-R1-Lift-Bin-IK-Abs-Direct-v0",
    entry_point="omni.isaac.lab_tasks.galaxea.direct.lift:R1LiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": R1LiftBinAbsEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FrankaCabinetPPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-R1-Lift-Bin-IK-Rel-Direct-v0",
    entry_point="omni.isaac.lab_tasks.galaxea.direct.lift:R1LiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": R1LiftBinRelEnvCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FrankaCabinetPPORunnerCfg,
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)



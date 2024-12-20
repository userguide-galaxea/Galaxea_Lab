# Copyright (c) 2024. Galaxea AI
# All rights reserved.


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Pick and lift state machine for lift environments."
)
parser.add_argument(
    "--cpu", action="store_true", default=False, help="Use CPU pipeline."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-R1-Multi-Fruit-IK-Abs-Direct-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=600,
    help="Number of episodes to store in the dataset.",
)
parser.add_argument(
    "--dataset_dir", type=str, default="dataset", help="Basename of output file."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import torch
import gymnasium as gym

import omni.isaac.lab_tasks  
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg                            
import os
import sys
import h5py
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rule_based_policy.utils.helper import compress_image_to_bytes
from state_machine.pick_and_place_sm import PickAndPlaceRightArmSm
import numpy as np
SIM_WARM_UP_STEP = 30

galaxea_datasets = {

            "/upper_body_observations/rgb_head":[], # bytes, png compressed image
            "/upper_body_observations/rgb_left_hand":[], # bytes, png compressed image
            "/upper_body_observations/rgb_right_hand":[], # bytes, png compressed image
            "/upper_body_observations/depth_head":[], # bytes, png compressed image
            "/upper_body_observations/depth_left_hand":[], # bytes, png compressed image
            "/upper_body_observations/depth_right_hand":[], # bytes, png compressed image
            "/upper_body_observations/left_arm_joint_position":[], # np.array(6)，fp32
            "/upper_body_observations/left_arm_joint_velocity":[], # np.array(6), fp32
            "/upper_body_observations/left_arm_gripper_position":[], # np.array(1), fp32
            "/upper_body_observations/left_arm_ee_torso_pose":[], # np.array(7),
    
            "/upper_body_observations/right_arm_joint_position":[], # np.array(6), fp32
            "/upper_body_observations/right_arm_joint_velocity":[], # np.array(6), fp32
            "/upper_body_observations/right_arm_gripper_position":[], # np.array(6), fp32
            "/upper_body_observations/right_arm_ee_torso_pose":[], # np.array(7), ee pose relative to torso_link4, position + quaternion

            "/lower_body_observations/torso_joint_position":[], # np.array(4) 
            "/lower_body_observations/left_arm_ee_base_pose":[], # np.array(7)  
            "/lower_body_observations/right_arm_ee_base_pose":[], # np.array(7)
            "/upper_body_action_dict/left_arm_gripper_position_cmd":[], # np.array(1)  
            "/upper_body_action_dict/left_arm_joint_position_cmd":[], # np.array(6)  
            "/upper_body_action_dict/left_arm_ee_torso_pose_cmd":[], # np.array(7)
            "/upper_body_action_dict/right_arm_gripper_position_cmd":[], # np.array(1)
            "/upper_body_action_dict/right_arm_joint_position_cmd":[], # np.array(6)
            "/upper_body_action_dict/right_arm_ee_torso_pose_cmd":[], # np.array(7)

            "/lower_body_action_dict/torso_joint_position_cmd":[], # np.array(4)
    }

def obs_to_dict(data_dict, obs, i):
    data_dict["/upper_body_observations/rgb_head"].append(compress_image_to_bytes(obs["front_rgb"][i]))
    data_dict["/upper_body_observations/rgb_left_hand"].append(compress_image_to_bytes(obs["left_rgb"][i]))
    data_dict["/upper_body_observations/rgb_right_hand"].append(compress_image_to_bytes(obs["right_rgb"][i]))
    data_dict["/upper_body_observations/depth_head"].append(compress_image_to_bytes(obs["front_depth"][i]))
    data_dict["/upper_body_observations/depth_left_hand"].append(compress_image_to_bytes(obs["left_depth"][i]))
    data_dict["/upper_body_observations/depth_right_hand"].append(compress_image_to_bytes(obs["right_depth"][i]))

    # left 
    data_dict["/upper_body_observations/left_arm_joint_position"].append(obs["joint_pos"][i][0:6])
    data_dict["/upper_body_observations/left_arm_joint_velocity"].append(obs["joint_vel"][i][0:6])
    data_dict["/upper_body_observations/left_arm_gripper_position"].append(obs["joint_pos"][i][6:7])
    data_dict["/upper_body_observations/left_arm_ee_torso_pose"].append(obs["left_ee_pose"][i])
    # right
    data_dict["/upper_body_observations/right_arm_joint_position"].append(obs["joint_pos"][i][7:13])
    data_dict["/upper_body_observations/right_arm_joint_velocity"].append(obs["joint_vel"][i][7:13])
    data_dict["/upper_body_observations/right_arm_gripper_position"].append(obs["joint_pos"][i][13:14])
    data_dict["/upper_body_observations/right_arm_ee_torso_pose"].append(obs["right_ee_pose"][i])

def actions_to_dict(data_dict, actions_record, i):
    data_dict["/upper_body_action_dict/left_arm_ee_torso_pose_cmd"].append(actions_record[i][0:7])
    data_dict["/upper_body_action_dict/left_arm_gripper_position_cmd"].append(actions_record[i][7:8])
    data_dict["/upper_body_action_dict/right_arm_ee_torso_pose_cmd"].append(actions_record[i][8:15])
    data_dict["/upper_body_action_dict/right_arm_gripper_position_cmd"].append(actions_record[i][15:16])

def save_demo(data_dict: dict, dataset_path: str, episode_idx: int, debug=False):
    if debug:
        for key, value in data_dict.items():
            print(f"key {key}, value shape: {len(value)}")

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    dataset_path = os.path.join(dataset_path, f"episode_{episode_idx}")

    if dataset_path is not None:
        with h5py.File(dataset_path + ".h5", "w", rdcc_nbytes=1024**2 * 2) as root:
            root.attrs["sim"] = True
            for key, value in data_dict.items():
                if isinstance(value, list) and all(isinstance(item, bytes) for item in value):
                #     # Handle list of bytes with np.void
                #     # Convert the list of bytes to a NumPy array of np.void
                    void_array = np.array(value, dtype=np.bytes_)
                    root.create_dataset(key, data=void_array)
                else:
                    root[key] = np.array(value)

            print(f"save file to {dataset_path}")


def reset_data_dict(data_dicts, index=None):
    if index is not None:
        for i in index:
            for key in galaxea_datasets.keys():
                data_dicts[i][key].clear()

def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        use_gpu=not args_cli.cpu,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # reset environment at start
    obs, _ = env.reset()
    print("init env...")

    sm = PickAndPlaceRightArmSm(
        args_cli.task,
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
    )

    data_dicts = [copy.deepcopy(galaxea_datasets) for _ in range(args_cli.num_envs)]

    count = 0
    episode_idx = 0
    episode_fail_idx = 0    
    flag = True 
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if count == 0:
                print(f"\nenter simulation loop...{episode_idx}")

            init_pos = env.unwrapped.init_pos
            task_id = env.unwrapped.object_id
            # get goal position
            goal_pos = obs["policy"]["goal_pose"][..., :3]
            # generate actions
            actions = sm.generate_actions(env, goal_pos)
            actions_record = actions.cpu().numpy() # fix the actions to save
            count += 1

            # give some time to warm up
            if count > SIM_WARM_UP_STEP:
                print(f"started! Episode {episode_idx}/{args_cli.num_demos}, step: {count}")
                # -- obs: process observations for saving
                obs = obs["policy"]
                obs = {k: v.cpu().numpy() for k, v in obs.items()}
                for i in range(args_cli.num_envs):
                    obs_to_dict(data_dicts[i], obs, i)

                # step environment
                obs, reward, terminated, truncated, info = env.step(actions)
                dones = terminated | truncated

            # -- set action
            if count > SIM_WARM_UP_STEP:
                for i in range(args_cli.num_envs):
                    actions_to_dict(data_dicts[i], actions_record, i)

                # reset state machine
                reset_idx = dones.nonzero(as_tuple=False).squeeze(-1)
                if dones.any():
                    print(
                        "terminated: {}, truncated: {}, reset_idx:{}".format(
                            terminated, truncated, reset_idx
                        )
                    )
                    for id in reset_idx:
                        if terminated[id]:
                            print(f"episode_{episode_idx}: reached the goal, save the demo")
                            save_demo(
                                data_dicts[id],
                                os.path.join(f"{args_cli.dataset_dir}", "task_"+str(task_id)),
                                episode_idx,
                                debug=True,
                            )
                            episode_idx += 1
                            np_saver_success = os.path.join(f"{args_cli.dataset_dir}", "task_"+str(task_id), "npy", "success")
                            if not os.path.exists(np_saver_success):
                                os.makedirs(np_saver_success)
                            np.save(os.path.join(np_saver_success, f"episode_{episode_idx}.npy"), init_pos[id].unsqueeze(0).cpu().numpy())
                        elif truncated[id]:  # truncated is True
                            episode_fail_idx += 1
                            np_saver_fail = os.path.join(f"{args_cli.dataset_dir}", "task_"+str(task_id), "npy", "fail")
                            if not os.path.exists(np_saver_fail):
                                os.makedirs(np_saver_fail)
                            np.save(os.path.join(np_saver_fail, f"episode_{episode_fail_idx}.npy"), init_pos[id].unsqueeze(0).cpu().numpy())
                            print(f"episode_{episode_fail_idx}: failed... reset the env")

                    # reset data_dict
                    reset_data_dict(data_dicts, reset_idx)
                    sm.reset_idx(reset_idx)


            if episode_idx >= args_cli.num_demos:
                break

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

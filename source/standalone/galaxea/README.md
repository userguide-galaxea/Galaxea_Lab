## Set NUCLEUS\_ASSET\_ROOT\_DIR

The path to 3D model assets is defined by NUCLEUS\_ASSET\_ROOT\_DIR in line 24 of `source/extensions/omni.isaac.lab/omni/isaac/lab/utils/assets.py` by default.
These assets are stored on AWS cloud by default, and loading them from within China is infeasible (and slow even with a VPN). Therefore, it is recommended to download all the 3D models yourself by following the instructions on this website: https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_faq.html#assets-pack. After downloading, set the NUCLEUS\_ASSET\_ROOT\_DIR to your local path:

```python
# in "source/extensions/omni.isaac.lab/omni/isaac/lab/utils/assets.py"
# NUCLEUS_ASSET_ROOT_DIR = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
USER_HOME_DIR = os.path.expanduser('~')
NUCLEUS_ASSET_ROOT_DIR = f"{USER_HOME_DIR}/App/nv/isaac-sim-assets-1-4.0.0/Assets/Isaac/4.0"
"""Path to the root directory on the Nucleus Server."""
```

## Usages:

### Basic

```bash
# spawn the R1 robot and set random joint positions
./isaaclab.sh -p source/standalone/galaxea/basic/spawn_robot.py

# create a scene consisting of object, table, robot and so on
./isaaclab.sh -p source/standalone/galaxea/basic/creat_scene.py

# simply create an env and call reset(), step(), you can also define the num_envs by modifying the code
./isaaclab.sh -p source/standalone/galaxea/basic/simple_env.py 

# test ik
./isaaclab.sh -p source/standalone/galaxea/basic/run_diff_ik.py

# follow the target with ik, select the cube prime and move it,
# the end-effector will track the cube
./isaaclab.sh -p source/standalone/galaxea/basic/follow_target_diff_ik.py



### Teleop

```bash
# control two arms with the keyboard asynchronously
./isaaclab.sh -p source/standalone/galaxea/teleop/teleop_biman_async.py --task Isaac-Lift-Cube-R1-IK-Rel-v0

# control two arms with the keyboard synchronously
./isaaclab.sh -p source/standalone/galaxea/teleop/teleop_biman_sync.py --task Isaac-Lift-Bin-R1-IK-Rel-v0
```

### Scripted Policy

```bash
# use DirectEnv workflow (preferred)
./isaaclab.sh -p source/standalone/galaxea/scripted_policy/lift.py --task Isaac-R1-Lift-Bin-IK-Abs-Direct-v0

./isaaclab.sh -p source/standalone/galaxea/scripted_policy/lift.py --task Isaac-R1-Lift-Cube-IK-Abs-Direct-v0

# use ManagerBasedEnv workflow (not being maintained)
./isaaclab.sh -p source/standalone/galaxea/scripted_policy/manager_based/lift_bin.py --num_envs 2

./isaaclab.sh -p source/standalone/galaxea/scripted_policy/manager_based/lift_cube.py
```



## Utils

* install PYTHON\_LIBS

```bash
/home/{user}/ws/xht_git_ws/isacc_lab_galaxea/_isaac_sim/kit/python/bin/python3 -m pip install PYTHON_LIBS
```

* find usd assets

  * https://build.nvidia.com/nvidia/usdsearch

* convert obj to usd

```bash
./isaaclab.sh -p source/standalone/tools/convert_mesh.py /home/{user}/ws/my_ws/assets/t_block/t_block.obj /home/{user}/ws/my_ws/assets/t_block/t.usd --headless --mass 0.03
```

* convert URDF to USD
  1. modify URDF: https://isaac-sim.github.io/IsaacLab/source/how-to/import\_new\_asset.html#example-usage,
  2. run (set **--fix-base** if you want to fix the base of the robot)

```bash
./isaaclab.sh -p source/standalone/tools/convert_urdf.py /home/{user}/ws/my_ws/assets/r1/r1.urdf /home/{user}/ws/my_ws/assets/r1/r1.usd --headless --fix-base
```

* to convert the R1.usd to an instanceable asset:
  * Open "Script Editor" from Window in Isaac Sim
  * Update "ASSET\_USD\_PATH" and "SAVE\_AS\_PATH" in create\_instanceable\_asset.py
  * Click "File", then click "Open", and choose the "create\_instanceable\_asset.py", then Click Run


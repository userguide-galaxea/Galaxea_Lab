import torch
from omni.isaac.lab.utils.math import quat_from_euler_xyz


def get_camera_rot_offset(
    roll: float, pitch: float = 0.0, yaw: float = -1.5708
) -> tuple:
    """Get the camera rotation offset in quaternion.
    default rotation offset with ROS convertion is (0.5, -0.5, 0.5, -0.5),
    normally, you only need to modify the roll angle.
    """
    rot_euler = torch.zeros(3)
    rot_euler[0] = -torch.pi + roll
    rot_euler[1] = pitch
    rot_euler[2] = yaw
    rot_quat = quat_from_euler_xyz(
        rot_euler[0],
        rot_euler[1],
        rot_euler[2],
    )
    return (rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3])

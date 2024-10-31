# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(3) control."""

import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform.rotation import Rotation

import carb
import omni

from ..device_base import DeviceBase


class Se3KeyboardBiman(DeviceBase):
    """A keyboard controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class is designed to provide a keyboard controller for a robotic arm with a gripper.
    It uses the Omniverse keyboard interface to listen to keyboard events and map them to robot's
    task-space commands.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Left:
          Toggle gripper (open/close)  B
          Move along x-axis            W                 S
          Move along y-axis            A                 D
          Move along z-axis            Q                 E
          Rotate along x-axis          Z                 X
          Rotate along y-axis          T                 G
          Rotate along z-axis          C                 V
        Right:
          Toggle gripper (open/close)  N
          Move along x-axis            I                 K
          Move along y-axis            J                 L
          Move along z-axis            U                 O
          Rotate along x-axis          M                 ,
          Rotate along y-axis          Y                 H
          Rotate along z-axis          .                 /
        ============================== ================= =================

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8):
        """Initialize the keyboard layer.

        Args:
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.05.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.5.
        """
        # store inputs
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(
                event, *args
            ),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # command buffers
        self._left_close_gripper = False
        self._right_close_gripper = False
        self._left_delta_pos = np.zeros(3)  # (x, y, z)
        self._left_delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        self._right_delta_pos = np.zeros(3)  # (x, y, z)
        self._right_delta_rot = np.zeros(3)  # (roll, pitch, yaw)

        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tToggle gripper (open/close): B(L), N(R)\n"
        msg += "\tMove arm along x-axis: W/S(L), I/K(R)\n"
        msg += "\tMove arm along y-axis: A/D(L), J/L(R)\n"
        msg += "\tMove arm along z-axis: Q/E(L), U/O(R)\n"
        msg += "\tRotate arm along x-axis: Z/X(L), M/,(R)\n"
        msg += "\tRotate arm along y-axis: T/G(L), Y/H(R)\n"
        msg += "\tRotate arm along z-axis: C/V(L), .//(R)\n"
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._left_close_gripper = False
        self._right_close_gripper = False
        self._left_delta_pos = np.zeros(3)  # (x, y, z)
        self._right_delta_pos = np.zeros(3)  # (x, y, z)
        self._left_delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        self._right_delta_rot = np.zeros(3)  # (roll, pitch, yaw)

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> tuple[np.ndarray, bool, np.ndarray, bool]:
        """Provides the result from keyboard event state.

        Returns:
            A tuple containing the delta pose command and gripper commands.
        """
        # convert to rotation vector
        left_rot_vec = Rotation.from_euler("XYZ", self._left_delta_rot).as_rotvec()
        right_rot_vec = Rotation.from_euler("XYZ", self._right_delta_rot).as_rotvec()
        # print("left_rot_vec: ", left_rot_vec)
        # print("right_rot_vec: ", right_rot_vec)
        # return the command and gripper state
        return (
            np.concatenate([self._left_delta_pos, left_rot_vec]),
            self._left_close_gripper,
            np.concatenate([self._right_delta_pos, right_rot_vec]),
            self._right_close_gripper,
        )

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset()
            elif event.input.name == "B":
                self._left_close_gripper = not self._left_close_gripper
            elif event.input.name == "N":
                self._right_close_gripper = not self._right_close_gripper
            elif event.input.name in ["W", "S", "Q", "E"]:
                self._left_delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["A", "D"]:
                self._left_delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["I", "K", "U", "O"]:
                self._right_delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["J", "L"]:
                self._right_delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._left_delta_rot += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["M", "COMMA", "Y", "H", "PERIOD", "SLASH"]:
                self._right_delta_rot += self._INPUT_KEY_MAPPING[event.input.name]
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["W", "S", "Q", "E"]:
                self._left_delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["A", "D"]:
                self._left_delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["I", "K", "U", "O"]:
                self._right_delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["J", "L"]:
                self._right_delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._left_delta_rot -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["M", "COMMA", "Y", "H", "PERIOD", "SLASH"]:
                self._right_delta_rot -= self._INPUT_KEY_MAPPING[event.input.name]
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # toggle: gripper command
            "B": True,
            "N": True,
            # x-axis (forward)
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "I": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "K": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            # y-axis (right-left)
            "D": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "A": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            "J": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "L": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            # z-axis (up-down)
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            "U": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "O": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            # roll (around x-axis)
            "Z": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "X": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "M": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "COMMA": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # pitch (around y-axis)
            "T": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            "Y": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "H": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            # yaw (around z-axis)
            "C": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
            "PERIOD": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "SLASH": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
        }

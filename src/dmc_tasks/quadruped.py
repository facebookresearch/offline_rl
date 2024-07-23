# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright 2019 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Quadruped Domain."""

import collections
import typing as tp
from typing import Any
import os

from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import xml_tools
from lxml import etree
import numpy as np

enums = mjbindings.enums
mjlib = mjbindings.mjlib


_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = .02

_WALK_SPEED = 0.5

_JUMP_HEIGHT = 1.0

# Named model elements.
_WALLS = ['wall_px', 'wall_py', 'wall_nx', 'wall_ny']

SUITE = containers.TaggedTasks()


def make(task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward: bool = False):
    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = task_kwargs.copy()
        task_kwargs['environment_kwargs'] = environment_kwargs
    env = SUITE[task](**task_kwargs)
    env.task.visualize_reward = visualize_reward
    return env


def make_model(floor_size=None, terrain: bool = False, rangefinders: bool = False,
               walls_and_ball: bool = False):
    """Returns the model XML string."""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    xml_string = common.read_model(os.path.join(root_dir, 'dmc_tasks', 'quadruped.xml'))
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    # Set floor size.
    if floor_size is not None:
        floor_geom = mjcf.find('.//geom[@name=\'floor\']')
        floor_geom.attrib['size'] = f'{floor_size} {floor_size} .5'

    # Remove walls, ball and target.
    if not walls_and_ball:
        for wall in _WALLS:
            wall_geom = xml_tools.find_element(mjcf, 'geom', wall)
            wall_geom.getparent().remove(wall_geom)

        # Remove ball.
        ball_body = xml_tools.find_element(mjcf, 'body', 'ball')
        ball_body.getparent().remove(ball_body)

        # Remove target.
        target_site = xml_tools.find_element(mjcf, 'site', 'target')
        target_site.getparent().remove(target_site)

    # Remove terrain.
    if not terrain:
        terrain_geom = xml_tools.find_element(mjcf, 'geom', 'terrain')
        terrain_geom.getparent().remove(terrain_geom)

    # Remove rangefinders if they're not used, as range computations can be
    # expensive, especially in a scene with heightfields.
    if not rangefinders:
        rangefinder_sensors = mjcf.findall('.//rangefinder')
        for rf in rangefinder_sensors:
            rf.getparent().remove(rf)

    return etree.tostring(mjcf, pretty_print=True)


@SUITE.add()
def stand(time_limit: int = _DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    xml_string = make_model(floor_size=_DEFAULT_TIME_LIMIT * _WALK_SPEED)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = Stand(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


@SUITE.add()
def jump(time_limit: int = _DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    xml_string = make_model(floor_size=_DEFAULT_TIME_LIMIT * _WALK_SPEED)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = Jump(desired_height=_JUMP_HEIGHT, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


# pylint: disable=attribute-defined-outside-init
class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Quadruped domain."""

    def _reload_from_data(self, data) -> None:
        super()._reload_from_data(data)
        # Clear cached sensor names when the physics is reloaded.
        self._sensor_types_to_names: tp.Dict[tp.Tuple[tp.Any, ...], tp.List[str]] = {}
        self._hinge_names: tp.List[str] = []
    
    def _get_sensor_names(self, *sensor_types) -> Any:
        try:
            sensor_names = self._sensor_types_to_names[sensor_types]
        except KeyError:
            [sensor_ids] = np.where(np.in1d(self.model.sensor_type, sensor_types))
            sensor_names = [self.model.id2name(s_id, 'sensor') for s_id in sensor_ids]
            self._sensor_types_to_names[sensor_types] = sensor_names
        return sensor_names

    def torso_upright(self) -> np.ndarray:
        """Returns the dot-product of the torso z-axis and the global z-axis."""
        return np.asarray(self.named.data.xmat['torso', 'zz'])

    def torso_velocity(self) -> Any:
        """Returns the velocity of the torso, in the local frame."""
        return self.named.data.sensordata['velocimeter'].copy()

    def com_height(self) -> Any:
        return self.named.data.sensordata['center_of_mass'].copy()[2]

    def egocentric_state(self) -> Any:
        """Returns the state without global orientation or position."""
        if not self._hinge_names:
            [hinge_ids] = np.nonzero(self.model.jnt_type ==
                                     enums.mjtJoint.mjJNT_HINGE)
            self._hinge_names = [self.model.id2name(j_id, 'joint')
                                 for j_id in hinge_ids]
        return np.hstack((self.named.data.qpos[self._hinge_names],
                          self.named.data.qvel[self._hinge_names],
                          self.data.act))

    def force_torque(self) -> Any:
        """Returns scaled force/torque sensor readings at the toes."""
        force_torque_sensors = self._get_sensor_names(enums.mjtSensor.mjSENS_FORCE,
                                                      enums.mjtSensor.mjSENS_TORQUE)
        return np.arcsinh(self.named.data.sensordata[force_torque_sensors])

    def imu(self) -> Any:
        """Returns IMU-like sensor readings."""
        imu_sensors = self._get_sensor_names(enums.mjtSensor.mjSENS_GYRO,
                                             enums.mjtSensor.mjSENS_ACCELEROMETER)
        return self.named.data.sensordata[imu_sensors]


def _find_non_contacting_height(physics, orientation, x_pos: float = 0.0, y_pos: float = 0.0) -> None:
    """Find a height with no contacts given a body orientation.
    Args:
      physics: An instance of `Physics`.
      orientation: A quaternion.
      x_pos: A float. Position along global x-axis.
      y_pos: A float. Position along global y-axis.
    Raises:
      RuntimeError: If a non-contacting configuration has not been found after
      10,000 attempts.
    """
    z_pos = 0.0  # Start embedded in the floor.
    num_contacts = 1
    num_attempts = 0
    # Move up in 1cm increments until no contacts.
    while num_contacts > 0:
        try:
            with physics.reset_context():
                physics.named.data.qpos['root'][:3] = x_pos, y_pos, z_pos
                physics.named.data.qpos['root'][3:] = orientation
        except control.PhysicsError:
            # We may encounter a PhysicsError here due to filling the contact
            # buffer, in which case we simply increment the height and continue.
            pass
        num_contacts = physics.data.ncon
        z_pos += 0.01
        num_attempts += 1
        if num_attempts > 10000:
            raise RuntimeError('Failed to find a non-contacting configuration.')


def _common_observations(physics) -> tp.Dict[str, Any]:
    """Returns the observations common to all tasks."""
    obs = collections.OrderedDict()
    obs['egocentric_state'] = physics.egocentric_state()
    obs['torso_velocity'] = physics.torso_velocity()
    obs['torso_upright'] = physics.torso_upright()
    obs['imu'] = physics.imu()
    obs['force_torque'] = physics.force_torque()
    return obs


def _upright_reward(physics, deviation_angle: int = 0):
    """Returns a reward proportional to how upright the torso is.
    Args:
      physics: an instance of `Physics`.
      deviation_angle: A float, in degrees. The reward is 0 when the torso is
        exactly upside-down and 1 when the torso's z-axis is less than
        `deviation_angle` away from the global z-axis.
    """
    deviation = np.cos(np.deg2rad(deviation_angle))
    return rewards.tolerance(
        physics.torso_upright(),
        bounds=(deviation, float('inf')),
        sigmoid='linear',
        margin=1 + deviation,
        value_at_margin=0)


class Stand(base.Task):
    def __init__(self, random=None) -> None:
        """Initializes an instance of `Move`.
        Args:
          desired_speed: A float. If this value is zero, reward is given simply
            for standing upright. Otherwise this specifies the horizontal velocity
            at which the velocity-dependent reward component is maximized.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super().__init__(random=random)

    def initialize_episode(self, physics) -> None:
        """Sets the state of the environment at the start of each episode.
        Args:
          physics: An instance of `Physics`.
        """
        # Initial configuration.
        orientation = self.random.randn(4)
        orientation /= np.linalg.norm(orientation)
        _find_non_contacting_height(physics, orientation)
        super().initialize_episode(physics)

    def get_observation(self, physics) -> tp.Dict[str, Any]:
        """Returns an observation to the agent."""
        return _common_observations(physics)

    def get_reward(self, physics) -> Any:
        """Returns a reward to the agent."""

        return _upright_reward(physics)


class Jump(base.Task):
    def __init__(self, desired_height, random=None) -> None:
        """Initializes an instance of `Move`.
        Args:
          desired_speed: A float. If this value is zero, reward is given simply
            for standing upright. Otherwise this specifies the horizontal velocity
            at which the velocity-dependent reward component is maximized.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._desired_height = desired_height
        super().__init__(random=random)

    def initialize_episode(self, physics) -> None:
        """Sets the state of the environment at the start of each episode.
        Args:
          physics: An instance of `Physics`.
        """
        # Initial configuration.
        orientation = self.random.randn(4)
        orientation /= np.linalg.norm(orientation)
        _find_non_contacting_height(physics, orientation)
        super().initialize_episode(physics)

    def get_observation(self, physics) -> tp.Dict[str, Any]:
        """Returns an observation to the agent."""
        return _common_observations(physics)

    def get_reward(self, physics) -> Any:
        """Returns a reward to the agent."""

        # Move reward term.
        jump_up = rewards.tolerance(physics.com_height(),
                                    bounds=(self._desired_height, float('inf')),
                                    margin=self._desired_height,
                                    value_at_margin=0.5,
                                    sigmoid='linear')

        return _upright_reward(physics) * jump_up
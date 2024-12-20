# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright 2017 The dm_control Authors.
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
"""Planar Walker Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Any, Tuple
import typing as tp
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources

_CONTROL_TIMESTEP: float
_DEFAULT_TIME_LIMIT: int
_RUN_SPEED: int
_STAND_HEIGHT: float

_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2

# Horizontal speed (meters/second) above which move reward is 1.
_RUN_SPEED = 8

# Threshold to define when a target speed is reached
_SPEED_THRESHOLD = 0.5
# The maximum allowed height for crawling
_CRAWL_HEIGHT = 0.4
# Minimum height to consider a foot off the ground
_FEET_HEIGHT = 0.2
# Maximum feet height allowed for crawling
_CRAWL_FEET_HEIGHT = 0.6
# Minimum angle allowed to be in a lie back position
_CRAWL_LIE_BACK = 0.9

# Angular velocity for the flip task
_SPIN_SPEED = 5

SUITE = containers.TaggedTasks()

def make(task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward: bool = False): # TODO remove?
    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = task_kwargs.copy()
        task_kwargs['environment_kwargs'] = environment_kwargs
    if task in SUITE:
        env = SUITE[task](**task_kwargs)
    elif task.startswith("run") or task.startswith("spin") or task.startswith("crawl"):
        env = create_locomotion(task, **task_kwargs)
    else:
        raise ValueError(f"Walker task {task} not found")
    env.task.visualize_reward = visualize_reward
    return env


def get_model_and_assets() -> Tuple[Any, Any]:
    """Returns a tuple containing the model XML string and a dict of assets."""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    xml = resources.GetResource(os.path.join(root_dir, 'dmc_tasks', 'walker.xml'))
    return xml, common.ASSETS

# NOTE we do not register locomotion environments to the DMC suite
def create_locomotion(task:str, time_limit: int = _DEFAULT_TIME_LIMIT,
                      random=None, environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    params = WalkerLocomotion.get_params(task)
    task = WalkerLocomotion(random=random, **params)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)

@SUITE.add('benchmarking')
def spin(time_limit: int = _DEFAULT_TIME_LIMIT,
         random=None,
         environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalker(move_speed=_RUN_SPEED,
                        forward=True,
                        spin=True,
                        random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)

####################################### CODE #######################################
class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Walker domain."""

    def torso_upright(self) -> Any:
        """Returns projection from z-axes of torso to the z-axes of world."""
        return self.named.data.xmat['torso', 'zz']

    def torso_height(self) -> Any:
        """Returns the height of the torso."""
        return self.named.data.xpos['torso', 'z']
    
    def horizontal_velocity(self) -> Any:
        """Returns the horizontal velocity of the center-of-mass."""
        return self.named.data.sensordata['torso_subtreelinvel'][0]

    def orientations(self) -> Any:
        """Returns planar orientations of all bodies."""
        return self.named.data.xmat[1:, ['xx', 'xz']].ravel()

    def angmomentum(self) -> Any:
        """Returns the angular momentum of torso of the Cheetah about Y axis."""
        return self.named.data.subtree_angmom['torso'][1]
    

class PlanarWalker(base.Task):
    """A planar walker task."""

    def __init__(self, move_speed, forward=True, spin=False, random=None) -> None:
        """Initializes an instance of `PlanarWalker`.

    Args:
      move_speed: A float. If this value is zero, reward is given simply for
        standing up. Otherwise this specifies a target horizontal velocity for
        the walking task.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
        self._move_speed = move_speed
        self._forward = 1 if forward else -1
        self._spin = spin
        super(PlanarWalker, self).__init__(random=random)

    def initialize_episode(self, physics) -> None:
        """Sets the state of the environment at the start of each episode.

    In 'standing' mode, use initial orientation and small velocities.
    In 'random' mode, randomize joint angles and let fall to the floor.

    Args:
      physics: An instance of `Physics`.
    """
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        super(PlanarWalker, self).initialize_episode(physics)

    def get_observation(self, physics) -> tp.Dict[str, Any]:
        """Returns an observation of body orientations, height and velocites."""
        obs = collections.OrderedDict()
        obs['orientations'] = physics.orientations()
        obs['height'] = physics.torso_height()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics) -> Any:
        """Returns a reward to the agent."""
        standing = rewards.tolerance(physics.torso_height(),
                                     bounds=(_STAND_HEIGHT, float('inf')),
                                     margin=_STAND_HEIGHT / 2)
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4
        if self._spin:
            spin_reward = rewards.tolerance(self._forward *
                                            physics.angmomentum(),
                                            bounds=(10, float('inf')),
                                            margin=10,
                                            value_at_margin=0,
                                            sigmoid='linear')
            return spin_reward

        move_reward = rewards.tolerance(self._forward * physics.horizontal_velocity(),
                                        bounds=(self._move_speed, float('inf')),
                                        margin=self._move_speed / 2,
                                        value_at_margin=0.5,
                                        sigmoid='linear')

        return stand_reward * (5 * move_reward + 1) / 6

#########################################################################
# WALKER ABSTRACT TASK
#########################################################################

class WalkerTask(base.Task):
    
    """Defined to collect reusable functionalities of Walker tasks"""
    def get_observation(self, physics) -> tp.Dict[str, Any]:
        """Returns an observation of body orientations, height and velocities."""
        obs = collections.OrderedDict()
        obs['orientations'] = physics.orientations()
        obs['height'] = physics.torso_height()
        obs['velocity'] = physics.velocity()
        return obs
    
    def initialize_episode(self, physics) -> None:
        """Sets the state of the environment at the start of each episode.

        In 'standing' mode, use initial orientation and small velocities.
        In 'random' mode, randomize joint angles and let fall to the floor.

        Args:
            physics: An instance of `Physics`.
        """
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        super(WalkerTask, self).initialize_episode(physics)


#########################################################################
# LOCOMOTION TASKS
#########################################################################
class WalkerLocomotion(WalkerTask):
    """Walker locomotion tasks that require the agent to move or rotate at a desired speed."""

    def __init__(self, speed=0, forward=True, mode="run", random=None,) -> None:
        """
        Args:
        speed: A float. If this value is zero, reward is given simply for
            standing up. Otherwise this specifies a target horizontal velocity.
        forward: whether the agent should move forward or backward.
        mode: it can be either ['run', 'spin', 'crawl']
        random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._speed = speed
        self._forward = 1 if forward else -1
        assert mode in ['run', 'spin', 'crawl'], "Walker locomotion mode supports only ['run', 'spin', 'crawl']"
        self._mode = mode
        super(WalkerLocomotion, self).__init__(random=random)

    @staticmethod
    def get_params(task:str):
        """Parse a string and return the parameters to build a WalkerLocomotion object"""
        if task.startswith("walker_"):
            _, task = task.split("_", maxsplit=1)
        mode, dr, speed = task.split("_", maxsplit=2)
        correct = mode in ['run', 'spin', 'crawl'] and dr in ['fw', 'bw'] and speed.startswith("speed") and len(speed) > 5
        assert correct, "wrong task format: must use walker_['run', 'spin', 'crawl']_['fw', 'bw']_speed[value]"
        return {"mode": mode, "speed": float(speed[5:]), "forward": dr == "fw"}

    def set_task(self, task: str):
        """Set the underlying task to the given one"""
        params = WalkerLocomotion.get_params(task)
        self._mode, self._speed, self._forward = params["mode"], params["speed"], params["forward"]
        self._forward = 1 if self._forward else -1

    def get_reward(self, physics) -> Any:
        """Returns a reward to the agent."""
        standing = rewards.tolerance(physics.torso_height(),
                            bounds=(_STAND_HEIGHT, float('inf')),
                            margin=_STAND_HEIGHT / 2)
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4

        # we simply return the stand task for zero speed
        if self._speed == 0:
            return stand_reward

        if self._mode == "spin":
            # reward is 1 if the angular momentum is within [_speed, _speed + _SPEED_THRESHOLD]
            # otherwise it decays linearly to 0 within _speed from the margins
            spin_reward = rewards.tolerance(self._forward * physics.angmomentum(),
                                            bounds=(self._speed, self._speed + _SPEED_THRESHOLD),
                                            margin=self._speed,
                                            value_at_margin=0,
                                            sigmoid='linear')
            return spin_reward

        assert self._mode in ["crawl", "run"]

        # reward is 1 if the horizontal speed is within [_speed, _speed + _SPEED_THRESHOLD]
        # otherwise it decays linearly to 0.5 within _speed / 2 from the margins
        move_reward = rewards.tolerance(self._forward * physics.horizontal_velocity(),
                                        bounds=(self._speed, self._speed + _SPEED_THRESHOLD),
                                        margin=self._speed / 2,
                                        value_at_margin=0.5,
                                        sigmoid='linear')

        if self._mode == "crawl":
            crawling = rewards.tolerance(physics.torso_height(),
                                         bounds=(0, _CRAWL_HEIGHT),
                                         margin=_CRAWL_HEIGHT / 2)
            lie_back = rewards.tolerance((physics.torso_orientation() + 1) / 2,
                                         bounds=(_CRAWL_LIE_BACK, float('inf')),
                                         margin=_CRAWL_LIE_BACK,
                                         value_at_margin=0,
                                         sigmoid='linear')
            crawl_reward = (3 * crawling + lie_back) / 4
            low_feet = sum([rewards.tolerance(foot_height,
                                               bounds=(0, _CRAWL_FEET_HEIGHT),
                                               margin=1-_CRAWL_FEET_HEIGHT,
                                               value_at_margin=0,
                                               sigmoid='linear') for foot_height in physics.feet_height()])
            return crawl_reward * (5 * move_reward + low_feet + 1) / 8

        # for run mode
        return stand_reward * (5 * move_reward + 1) / 6
#########################################################################

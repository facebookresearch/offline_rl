# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
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
"""Point-mass domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources
from dm_env import specs
import numpy as np
import os

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()

TASKS = [('reach_top_left', np.array([-0.15, 0.15])),
         ('reach_top_right', np.array([0.15, 0.15])),
         ('reach_bottom_left', np.array([-0.15, -0.15])),
         ('reach_bottom_right', np.array([0.15, -0.15])),
         ('reach_bottom_left_long', np.array([-0.15, -0.15])),
         ('loop', None),
         ('square', None),
         ('fast_slow', None)]


def make(task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward=False):
    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = task_kwargs.copy()
        task_kwargs['environment_kwargs'] = environment_kwargs
    env = SUITE[task](**task_kwargs)
    env.task.visualize_reward = visualize_reward
    return env


def get_model_and_assets(task):
    """Returns a tuple containing the model XML string and a dict of assets."""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    xml = resources.GetResource(
        os.path.join(root_dir, 'dmc_tasks', f'pointmass_{task}.xml'))
    return xml, common.ASSETS


@SUITE.add('benchmarking')
def reach_top_left(time_limit=_DEFAULT_TIME_LIMIT,
                   random=None,
                   environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets('reach_top_left'))
    task = PointMassMaze(target_id=0, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)

@SUITE.add('benchmarking')
def loop(time_limit=_DEFAULT_TIME_LIMIT,
                   random=None,
                   environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets('nogoal'))
    task = PointMassMaze(target_id=5, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)

@SUITE.add('benchmarking')
def square(time_limit=_DEFAULT_TIME_LIMIT,
                   random=None,
                   environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets('nogoal'))
    task = PointMassMaze(target_id=6, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)

@SUITE.add('benchmarking')
def fast_slow(time_limit=_DEFAULT_TIME_LIMIT,
                   random=None,
                   environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets('nogoal'))
    task = PointMassMaze(target_id=7, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)

@SUITE.add('benchmarking')
def reach_top_right(time_limit=_DEFAULT_TIME_LIMIT,
                    random=None,
                    environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets('reach_top_right'))
    task = PointMassMaze(target_id=1, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


@SUITE.add('benchmarking')
def reach_bottom_left(time_limit=_DEFAULT_TIME_LIMIT,
                      random=None,
                      environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets('reach_bottom_left'))
    task = PointMassMaze(target_id=2, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)

@SUITE.add('benchmarking')
def reach_bottom_left_long(time_limit=_DEFAULT_TIME_LIMIT,
                      random=None,
                      environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets('reach_bottom_left'))
    task = PointMassMaze(target_id=4, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


@SUITE.add('benchmarking')
def reach_bottom_right(time_limit=_DEFAULT_TIME_LIMIT,
                       random=None,
                       environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets('reach_bottom_right'))
    task = PointMassMaze(target_id=3, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


class Physics(mujoco.Physics):
    """physics for the point_mass domain."""

    def mass_to_target_dist(self, target):
        """Returns the distance from mass to the target."""
        d = target - self.named.data.geom_xpos['pointmass'][:2]
        return np.linalg.norm(d)


class PointMassMaze(base.Task):
    """A point_mass `Task` to reach target with smooth reward."""

    def __init__(self, target_id, random=None) -> None:
        """Initialize an instance of `PointMassMaze`.

            Args:
                random: Optional, either a `numpy.random.RandomState` instance, an
                integer seed for creating a new `RandomState`, or None to select a seed
                automatically (default).
        """
        self._target = TASKS[target_id][1]
        self._id = target_id
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

            Args:
                physics: An instance of `mujoco.Physics`.
        """
        randomizers.randomize_limited_and_rotational_joints(
            physics, self.random)
        physics.data.qpos[0] = np.random.uniform(-0.29, -0.15)
        physics.data.qpos[1] = np.random.uniform(0.15, 0.29)
        # import ipdb; ipdb.set_trace()
        physics.named.data.geom_xpos['target'][:2] = self._target

        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward_spec(self):
        return specs.Array(shape=(1,), dtype=np.float32, name='reward')

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        x, y = physics.position()
        vx, vy = physics.velocity()
        
        if self._id == 7:  # fast_slow
            up = int(y > 0.2 and y < 0.28)
            right = int(x > 0.2 and x < 0.28)
            left = int(x < -0.2 and x > -0.28)
            down = int(y < -0.2 and y > -0.28)

            up_rew = rewards.tolerance(vx, bounds=(-0.05, -0.04), margin=0.01, value_at_margin=0, sigmoid='linear') * up
            right_rew = rewards.tolerance(vy, bounds=(0.09, 0.1), margin=0.01, value_at_margin=0, sigmoid='linear') * right
            left_rew = rewards.tolerance(vy, bounds=(-0.1, -0.09), margin=0.01, value_at_margin=0, sigmoid='linear') * left
            down_rew = rewards.tolerance(vx, bounds=(0.04, 0.05), margin=0.01, value_at_margin=0, sigmoid='linear') * down

            reward = 0 if up + right + left + down > 1 else up_rew + right_rew + left_rew + down_rew
        elif self._id == 6:  # square
            up = int(y > 0.2)
            right = int(x > 0.2)
            left = int(x < -0.2)
            down = int(y < -0.2)

            up_rew = abs(np.clip(vx, 0, 0.1) * 10 * up)
            right_rew = abs(np.clip(vy, -0.1, 0) * 10 * right)
            left_rew = abs(np.clip(vy, 0, 0.1) * 10 * left)
            down_rew = abs(np.clip(vx, -0.1, 0) * 10 * down)

            reward = 0 if up + right + left + down > 1 else up_rew + right_rew + left_rew + down_rew
        elif self._id == 5:  # loop
            tl = x <= 0 and y >= 0
            tr = x > 0 and y >= 0
            bl = x <= 0 and y < 0
            br = x > 0 and y < 0

            if tl:
                vx_rew = rewards.tolerance(vx, bounds=(0.06, 0.1), margin=0.01, value_at_margin=0, sigmoid='linear')
                vy_rew = rewards.tolerance(vy, bounds=(0.06, 0.1), margin=0.01, value_at_margin=0, sigmoid='linear')
                a, b, c = 1, -1, 0.24
            elif tr:
                vx_rew = rewards.tolerance(vx, bounds=(0.06, 0.1), margin=0.01, value_at_margin=0, sigmoid='linear')
                vy_rew = rewards.tolerance(vy, bounds=(-0.1, -0.06), margin=0.01, value_at_margin=0, sigmoid='linear')
                a, b, c = -1, -1, 0.24
            elif bl:
                vx_rew = rewards.tolerance(vx, bounds=(-0.1, -0.06), margin=0.01, value_at_margin=0, sigmoid='linear')
                vy_rew = rewards.tolerance(vy, bounds=(0.06, 0.1), margin=0.01, value_at_margin=0, sigmoid='linear')
                a, b, c = -1, -1, -0.24
            elif br:
                vx_rew = rewards.tolerance(vx, bounds=(-0.1, -0.06), margin=0.01, value_at_margin=0, sigmoid='linear')
                vy_rew = rewards.tolerance(vy, bounds=(-0.1, -0.06), margin=0.01, value_at_margin=0, sigmoid='linear')
                a, b, c = 1, -1, -0.24
            else:
                raise Exception()

            dist = abs(a*x + b*y + c) / np.sqrt(2)
            dist_rew = rewards.tolerance(dist, bounds=(0, 0.02), margin=0.02, value_at_margin=0, sigmoid='linear')
            reward = (dist_rew + vx_rew + vy_rew) / 3
        else:
            target_size = .015
            control_reward = rewards.tolerance(physics.control(), margin=1,
                                            value_at_margin=0,
                                            sigmoid='quadratic').mean()
            small_control = (control_reward + 4) / 5
            near_target = rewards.tolerance(physics.mass_to_target_dist(self._target),
                                            bounds=(0, target_size), margin=6 * target_size)

            reward = near_target * small_control
            if self._id == 4:  # reach_bottom_left_long
                if reward < 0.01:
                    up = int(y > 0.15)
                    right = int(x > 0.15)
                    left = int(x < -0.15)
                    down = int(y < -0.15)
                    up_rew = np.clip(vx, -0.1, 0.1) * up * (5 if vx >= 0 else 100)
                    right_rew = -np.clip(vy, -0.1, 0.1) * right * (5 if vy <= 0 else 100)
                    left_rew = np.clip(vy, -0.1, 0.1) * left * (5 if vy >= 0 else 100)
                    down_rew = -np.clip(vx, -0.1, 0.1) * down * (5 if vx <= 0 else 100)
                    reward = 0 if up + right + left + down > 1 else up_rew + right_rew + left_rew + down_rew
        return reward

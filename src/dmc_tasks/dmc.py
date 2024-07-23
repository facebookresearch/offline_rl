# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import dataclasses
import typing as tp
from typing import Any
import numpy as np

from dm_env import Environment
from dm_env import StepType, specs
# pytlint: disable=import-outside-toplevel

from . import walker
from . import cheetah
from . import quadruped
from . import humanoid

class UnsupportedPlatform(unittest.SkipTest, RuntimeError):
    """The platform is not supported for running"""

try:
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale
except ImportError as e:
    raise UnsupportedPlatform(f"Import error (Note: DMC does not run on Mac):\n{e}") from e


S = tp.TypeVar("S", bound="TimeStep")
Env = tp.Union["EnvWrapper", Environment]


@dataclasses.dataclass
class TimeStep:
    step_type: StepType
    reward: float
    discount: float
    observation: np.ndarray
    physics: np.ndarray = dataclasses.field(default=np.ndarray([]), init=False)

    def first(self) -> bool:
        return self.step_type == StepType.FIRST  # type: ignore

    def mid(self) -> bool:
        return self.step_type == StepType.MID  # type: ignore

    def last(self) -> bool:
        return self.step_type == StepType.LAST  # type: ignore

    def __getitem__(self, attr: str) -> tp.Any:
        return getattr(self, attr)

    def _replace(self: S, **kwargs: tp.Any) -> S:
        for name, val in kwargs.items():
            setattr(self, name, val)
        return self


@dataclasses.dataclass
class ExtendedTimeStep(TimeStep):
    action: tp.Any


class EnvWrapper:
    def __init__(self, env: Env) -> None:
        self._env = env

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        if not isinstance(time_step, TimeStep):
            # dm_env time step is a named tuple
            time_step = TimeStep(**time_step._asdict())
        if self.physics is not None:
            return time_step._replace(physics=self.physics.get_state())
        else:
            return time_step

    def reset(self) -> TimeStep:
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action: np.ndarray) -> TimeStep:
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def observation_spec(self) -> tp.Any:
        assert isinstance(self, EnvWrapper)
        return self._env.observation_spec()

    def action_spec(self) -> specs.Array:
        return self._env.action_spec()

    def render(self, *args: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        return self._env.render(*args, **kwargs)  # type: ignore

    def close(self) -> None:
        self.base_env.close()

    @property
    def base_env(self) -> tp.Any:
        env = self._env
        if isinstance(env, EnvWrapper):
            return self._env.base_env
        return env

    @property
    def physics(self) -> tp.Any:
        if hasattr(self._env, "physics"):
            return self._env.physics

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(EnvWrapper):
    def __init__(self, env: Env, dtype) -> None:
        super().__init__(env)
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def action_spec(self) -> specs.BoundedArray:
        return self._action_spec

    def step(self, action) -> Any:
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)


class ObservationDTypeWrapper(EnvWrapper):
    def __init__(self, env: Env, dtype) -> None:
        super().__init__(env)
        self._dtype = dtype
        wrapped_obs_spec = env.observation_spec()['observations']
        self._obs_spec = specs.Array(wrapped_obs_spec.shape, dtype,
                                     'observation')

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        obs = time_step.observation['observations'].astype(self._dtype)
        return time_step._replace(observation=obs)

    def observation_spec(self) -> Any:
        return self._obs_spec


class ExtendedTimeStepWrapper(EnvWrapper):

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        ts = ExtendedTimeStep(observation=time_step.observation,
                              step_type=time_step.step_type,
                              action=action,
                              reward=time_step.reward or 0.0,
                              discount=time_step.discount or 1.0)
        return super()._augment_time_step(time_step=ts, action=action)


# TODO import other domains
def _make_dmc(domain, task, seed):
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs=dict(random=seed),
                         environment_kwargs=dict(flat_observation=True),
                         visualize_reward=False)
    elif domain == 'walker':
        return walker.make(task,
                           task_kwargs=dict(random=seed),
                           environment_kwargs=dict(flat_observation=True),
                           visualize_reward=False)
    elif domain == 'cheetah':
        return cheetah.make(task,
                            task_kwargs=dict(random=seed),
                            environment_kwargs=dict(flat_observation=True),
                            visualize_reward=False)
    elif domain == 'humanoid':
        return humanoid.make(task,
                             task_kwargs=dict(random=seed),
                             environment_kwargs=dict(flat_observation=True),
                             visualize_reward=False)
    elif domain == 'quadruped':
        return quadruped.make(task,
                              task_kwargs=dict(random=seed),
                              environment_kwargs=dict(flat_observation=True),
                              visualize_reward=False)
    else:
        raise ValueError(f'{task} not found')

    return ActionDTypeWrapper(env, np.float32)


def make(name: str, seed=1) -> EnvWrapper:
    domain, task = name.split('_', 1)
    env = _make_dmc(domain, task, seed)  # type: ignore
    env = ObservationDTypeWrapper(env, np.float32)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = ExtendedTimeStepWrapper(env)
    return env

# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Environments for training and evaluating policies."""

import functools
from typing import Optional, Type

from brax.envs import ant2
from brax.envs import fast
from brax.envs import hopper2
from brax.envs import walker2d2
from brax.envs.base import Env, PipelineEnv, State, Wrapper
from brax.envs.wrappers import training

_envs = {
    'ant2': ant2.Ant2,
    'fast': fast.Fast,
    'hopper2': hopper2.Hopper2,
    'walker2d2': walker2d2.Walker2d2,
}



def get_environment(env_name: str, **kwargs) -> Env:
  """Returns an environment from the environment registry.

  Args:
    env_name: environment name string
    **kwargs: keyword arguments that get passed to the Env class constructor

  Returns:
    env: an environment
  """
  return _envs[env_name](**kwargs)


def register_environment(env_name: str, env_class: Type[Env]):
  """Adds an environment to the registry.

  Args:
    env_name: environment name string
    env_class: the Env class to add to the registry
  """
  _envs[env_name] = env_class


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    **kwargs,
) -> Env:
  """Creates an environment from the registry.

  Args:
    env_name: environment name string
    episode_length: length of episode
    action_repeat: how many repeated actions to take per environment step
    auto_reset: whether to auto reset the environment after an episode is done
    batch_size: the number of environments to batch together
    **kwargs: keyword argments that get passed to the Env class constructor

  Returns:
    env: an environment
  """
  env = _envs[env_name](**kwargs)

  if episode_length is not None:
    env = training.EpisodeWrapper(env, episode_length, action_repeat)
  if batch_size:
    env = training.VmapWrapper(env, batch_size)
  if auto_reset:
    env = training.AutoResetWrapper(env)

  return env

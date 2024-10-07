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
"""Wrappers to support Brax training."""

from typing import Dict, Optional

from brax.envs.base import Env, State, Wrapper
from flax import struct
import jax
from jax import numpy as jp


def wrap(env: Env, episode_length: int = 1000,
         action_repeat: int = 1, obs_history_length: int = 1,
         reset_at_episode_end: bool = True) -> Wrapper:
  """Common wrapper pattern for all training agents.

  Args:
    env: environment to be wrapped episode_length: length of episode
    action_repeat: how many repeated actions to take per step
    obs_history_length: how many obs frames to stack together as a single observation
    reset_at_episode_end: whether to reset the environment when the episode ends

  Returns:
    An environment that is wrapped with FrameStack, Episode and AutoReset
    wrappers.  If the environment did not already have batch dimensions, it is
    additional Vmap wrapped.
  """

  env = FrameStackWrapper(env, obs_history_length)
  env = EpisodeWrapper(env, episode_length, action_repeat, reset_at_episode_end)
  env = VmapWrapper(env)
  env = AutoResetWrapper(env)
  return env


class FrameStackWrapper(Wrapper):
  """Maintains frame stack of observations.
  
  Newest observations are at the beginning of the observation vector.
  
  """

  def __init__(self, env: Env, obs_history_length: int):
    super().__init__(env)
    self.obs_history_length = obs_history_length
    self._unwrapped_obs_size = self.unwrapped.observation_size

  def reset(self, rng: jp.ndarray) -> State:
    nstate = self.env.reset(rng)
    obs_stack = jp.tile(nstate.obs, self.obs_history_length)
    return nstate.replace(obs=obs_stack, prev_obs=obs_stack)

  def step(self, state: State, action: jp.ndarray) -> State:
    obs_old_stack = state.obs
    obs_old = state.obs[:self._unwrapped_obs_size]
    nstate = self.env.step(state.replace(obs=obs_old), action)
    obs_stack = jp.concatenate(
      [nstate.obs, state.obs[:self._unwrapped_obs_size*(self.obs_history_length-1)]],
      axis=-1
    )
    return nstate.replace(obs=obs_stack, prev_obs=obs_old_stack)

  def approx_dynamics(self, obs: jp.ndarray, u: jp.ndarray,
                      ext_forces: Optional[jp.ndarray] = None,
                      obs_next: Optional[jp.ndarray] = None) -> State:
    obs_old = obs[:self._unwrapped_obs_size]
    obs_next_unstacked = obs_next[:self._unwrapped_obs_size]
    new_obs = self.env.approx_dynamics(obs_old, u, ext_forces, obs_next_unstacked)
    obs_stack = jp.concatenate(
      [new_obs, obs[:self._unwrapped_obs_size*(self.obs_history_length-1)]],
      axis=-1
    )
    return obs_stack

  def low_level_control(self, action: jp.ndarray,
                        obs: jp.ndarray) -> jp.ndarray:
    return self.env.low_level_control(action, obs[:self._unwrapped_obs_size])

  def compute_reward(self, obs: jp.ndarray, prev_obs: jp.ndarray,
                     u: jp.ndarray, unscaled_action: jp.ndarray) -> jp.ndarray:
    return self.env.compute_reward(obs[:self._unwrapped_obs_size],
                                   prev_obs[:self._unwrapped_obs_size],
                                   u, unscaled_action)

  @property
  def observation_size(self) -> int:
    return self._unwrapped_obs_size * self.frame_stack

class VmapWrapper(Wrapper):
  """Vectorizes Brax env."""

  def __init__(self, env: Env, batch_size: Optional[int] = None):
    super().__init__(env)
    self.batch_size = batch_size

  def reset(self, rng: jp.ndarray) -> State:
    if self.batch_size is not None:
      rng = jax.random.split(rng, self.batch_size)
    return jax.vmap(self.env.reset)(rng)

  def step(self, state: State, action: jp.ndarray) -> State:
    return jax.vmap(self.env.step)(state, action)

  def approx_dynamics(self, obs: jp.ndarray, u: jp.ndarray,
                      ext_forces: Optional[jp.ndarray] = None,
                      obs_next: Optional[jp.ndarray] = None) -> State:
    return jax.vmap(self.env.approx_dynamics)(obs, u, ext_forces, obs_next)

  def low_level_control(self, action: jp.ndarray,
                        obs: jp.ndarray) -> jp.ndarray:
    return jax.vmap(self.env.low_level_control)(action, obs)

  def compute_reward(self, obs: jp.ndarray, prev_obs: jp.ndarray,
                     u: jp.ndarray, unscaled_action: jp.ndarray) -> jp.ndarray:
    return jax.vmap(self.env.compute_reward)(obs, prev_obs, u, unscaled_action)


class EpisodeWrapper(Wrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: Env, episode_length: int, action_repeat: int,
               set_done_at_end: bool = True):
    super().__init__(env)
    self.episode_length = episode_length
    self.action_repeat = action_repeat
    self.set_done_at_end = set_done_at_end

  def reset(self, rng: jp.ndarray) -> State:
    state = self.env.reset(rng)
    state.info['steps'] = jp.zeros(rng.shape[:-1])
    state.info['truncation'] = jp.zeros(rng.shape[:-1])
    return state

  def step(self, state: State, action: jp.ndarray) -> State:
    def f(state, _):
      nstate = self.env.step(state, action)
      return nstate, nstate.reward

    state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
    state = state.replace(reward=jp.sum(rewards, axis=0))
    steps = state.info['steps'] + self.action_repeat
    one = jp.ones_like(state.done)
    zero = jp.zeros_like(state.done)
    episode_length = jp.array(self.episode_length, dtype=jp.int32)
    done = jp.where((steps >= episode_length) & self.set_done_at_end, one, state.done)
    state.info['truncation'] = jp.where(
        steps >= episode_length, 1 - state.done, zero
    )
    state.info['steps'] = steps
    return state.replace(done=done)


class AutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jp.ndarray) -> State:
    state = self.env.reset(rng)
    state.info['first_pipeline_state'] = state.pipeline_state
    state.info['first_obs'] = state.obs
    state.info['first_metrics'] = state.metrics
    return state

  def step(self, state: State, action: jp.ndarray) -> State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
      return jp.where(done, x, y)

    pipeline_state = jax.tree_map(
        where_done, state.info['first_pipeline_state'], state.pipeline_state
    )
    obs = where_done(state.info['first_obs'], state.obs)
    metrics = jax.tree_map(
        where_done, state.info['first_metrics'], state.metrics
    )
    return state.replace(
      pipeline_state=pipeline_state, obs=obs, metrics=metrics
    )


@struct.dataclass
class EvalMetrics:
  """Dataclass holding evaluation metrics for Brax.

  Attributes:
      episode_metrics: Aggregated episode metrics since the beginning of the
        episode.
      active_episodes: Boolean vector tracking which episodes are not done yet.
      episode_steps: Integer vector tracking the number of steps in the episode.
  """

  episode_metrics: Dict[str, jp.ndarray]
  active_episodes: jp.ndarray
  episode_steps: jp.ndarray


class EvalWrapper(Wrapper):
  """Brax env with eval metrics."""

  def reset(self, rng: jp.ndarray) -> State:
    reset_state = self.env.reset(rng)
    reset_state.metrics['reward'] = reset_state.reward
    eval_metrics = EvalMetrics(
        episode_metrics=jax.tree_util.tree_map(
            jp.zeros_like, reset_state.metrics
        ),
        active_episodes=jp.ones_like(reset_state.reward),
        episode_steps=jp.zeros_like(reset_state.reward),
    )
    reset_state.info['eval_metrics'] = eval_metrics
    return reset_state

  def step(self, state: State, action: jp.ndarray) -> State:
    state_metrics = state.info['eval_metrics']
    if not isinstance(state_metrics, EvalMetrics):
      raise ValueError(
          f'Incorrect type for state_metrics: {type(state_metrics)}'
      )
    del state.info['eval_metrics']
    nstate = self.env.step(state, action)
    nstate.metrics['reward'] = nstate.reward
    episode_steps = jp.where(
        state_metrics.active_episodes,
        nstate.info['steps'],
        state_metrics.episode_steps,
    )
    episode_metrics = jax.tree_util.tree_map(
        lambda a, b: a + b * state_metrics.active_episodes,
        state_metrics.episode_metrics,
        nstate.metrics,
    )
    active_episodes = state_metrics.active_episodes * (1 - nstate.done)

    eval_metrics = EvalMetrics(
        episode_metrics=episode_metrics,
        active_episodes=active_episodes,
        episode_steps=episode_steps,
    )
    nstate.info['eval_metrics'] = eval_metrics
    return nstate

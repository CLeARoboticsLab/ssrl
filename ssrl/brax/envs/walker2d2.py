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
"""Trains a 2D walker to run in the +x direction."""

from typing import Tuple, Any

from brax import base
from brax.envs.base import PipelineEnv, State
from brax import actuator
from brax import kinematics
from brax.generalized.base import State as GeneralizedState
from brax.generalized import dynamics
from brax.generalized import integrator
from brax.generalized import mass
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp


class Walker2d2(PipelineEnv):



  # pyformat: disable
  """
  ### Description

  Same as Walker2d, but with a separated out function for determining dones.

  This environment builds on the hopper environment based on the work done by
  Erez, Tassa, and Todorov in
  ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"](http://www.roboticsproceedings.org/rss07/p10.pdf)
  by adding another set of legs making it possible for the robot to walker
  forward instead of hop. Like other Mujoco environments, this environment aims
  to increase the number of independent state and control variables as compared
  to the classic control environments.

  The walker is a two-dimensional two-legged figure that consist of four main
  body parts - a single torso at the top (with the two legs splitting after the
  torso), two thighs in the middle below the torso, two legs in the bottom below
  the thighs, and two feet attached to the legs on which the entire body rests.

  The goal is to make coordinate both sets of feet, legs, and thighs to move in
  the forward (right) direction by applying torques on the six hinges connecting
  the six body parts.

  ### Action Space

  The agent take a 6-element vector for actions. The action space is a
  continuous `(action, action, action, action, action, action)` all in
  `[-1, 1]`, where `action` represents the numerical torques applied at the
  hinge joints.

  | Num | Action                                 | Control Min | Control Max | Name (in corresponding config) | Joint | Unit         |
  |-----|----------------------------------------|-------------|-------------|--------------------------------|-------|--------------|
  | 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                    | hinge | torque (N m) |
  | 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                      | hinge | torque (N m) |
  | 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                     | hinge | torque (N m) |
  | 3   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint               | hinge | torque (N m) |
  | 4   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                 | hinge | torque (N m) |
  | 5   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                | hinge | torque (N m) |

  ### Observation Space

  The state space consists of positional values of different body parts of the
  walker, followed by the velocities of those individual parts (their
  derivatives) with all the positions ordered before all the velocities.

  The observation is a `ndarray` with shape `(17,)` where the elements
  correspond to the following:

  | Num | Observation                                      | Min  | Max | Name (in corresponding config) | Joint | Unit                     |
  |-----|--------------------------------------------------|------|-----|--------------------------------|-------|--------------------------|
  | 0   | z-coordinate of the top (height of hopper)       | -Inf | Inf | rootz (torso)                  | slide | position (m)             |
  | 1   | angle of the top                                 | -Inf | Inf | rooty (torso)                  | hinge | angle (rad)              |
  | 2   | angle of the thigh joint                         | -Inf | Inf | thigh_joint                    | hinge | angle (rad)              |
  | 3   | angle of the leg joint                           | -Inf | Inf | leg_joint                      | hinge | angle (rad)              |
  | 4   | angle of the foot joint                          | -Inf | Inf | foot_joint                     | hinge | angle (rad)              |
  | 5   | angle of the left thigh joint                    | -Inf | Inf | thigh_left_joint               | hinge | angle (rad)              |
  | 6   | angle of the left leg joint                      | -Inf | Inf | leg_left_joint                 | hinge | angle (rad)              |
  | 7   | angle of the left foot joint                     | -Inf | Inf | foot_left_joint                | hinge | angle (rad)              |
  | 8   | velocity of the x-coordinate of the top          | -Inf | Inf | rootx                          | slide | velocity (m/s)           |
  | 9   | velocity of the z-coordinate (height) of the top | -Inf | Inf | rootz                          | slide | velocity (m/s)           |
  | 10  | angular velocity of the angle of the top         | -Inf | Inf | rooty                          | hinge | angular velocity (rad/s) |
  | 11  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_joint                    | hinge | angular velocity (rad/s) |
  | 12  | angular velocity of the leg hinge                | -Inf | Inf | leg_joint                      | hinge | angular velocity (rad/s) |
  | 13  | angular velocity of the foot hinge               | -Inf | Inf | foot_joint                     | hinge | angular velocity (rad/s) |
  | 14  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_left_joint               | hinge | angular velocity (rad/s) |
  | 15  | angular velocity of the leg hinge                | -Inf | Inf | leg_left_joint                 | hinge | angular velocity (rad/s) |
  | 16  | angular velocity of the foot hinge               | -Inf | Inf | foot_left_joint                | hinge | angular velocity (rad/s) |

  ### Rewards

  The reward consists of three parts:
  - *reward_healthy*: Every timestep that the walker is alive, it gets a reward of
    1
  - *reward_forward*: A reward of walking forward which is measured as
    *(x-coordinate before action - x-coordinate after action) / dt*.  *dt* is
    the time between actions - the default *dt = 0.008*. This reward would be
    positive if the walker walks forward (right) desired.
  - *reward_ctrl*: A negative reward for penalising the walker if it takes
    actions that are too large. It is measured as *-coefficient **x**
    sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
    control and has a default value of 0.001

  ### Starting State

  All observations start in state (0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) with a uniform noise in the range of
  [-0.005, 0.005] added to the values for stochasticity.

  ### Episode Termination

  The episode terminates when any of the following happens:

  1. The episode duration reaches a 1000 timesteps
  2. The height of the walker is ***not*** in the range `[0.7, 2]`
  3. The absolute value of the angle is ***not*** in the range `[-1, 1]`
  """
  # pyformat: enable


  def __init__(
      self,
      forward_reward_weight: float = 1.0,
      ctrl_cost_weight: float = 1e-3,
      healthy_reward: float = 1.0,
      terminate_when_unhealthy: bool = True,
      healthy_z_range: Tuple[float, float] = (0.8, 2.0),
      healthy_angle_range=(-1.0, 1.0),
      reset_noise_scale=5e-3,
      exclude_current_positions_from_observation=True,
      backend='generalized',
      **kwargs
  ):
    """Creates a Walker environment.

    Args:
      forward_reward_weight: Weight for the forward reward, i.e. velocity in
        x-direction.
      ctrl_cost_weight: Weight for the control cost.
      healthy_reward: Reward for staying healthy, i.e. respecting the posture
        constraints.
      terminate_when_unhealthy: Done bit will be set when unhealthy if true.
      healthy_z_range: Range of the z-position for being healthy.
      healthy_angle_range: Range of joint angles for being healthy.
      reset_noise_scale: Scale of noise to add to reset states.
      exclude_current_positions_from_observation: x-position will be hidden from
        the observations if true.
      backend: str, the physics backend to use
      **kwargs: Arguments that are passed to the base class.
    """
    path = epath.resource_path('brax') / 'envs/assets/walker2d.xml'
    sys = mjcf.load(path)
    sys = sys.replace(matrix_inv_iterations=0)

    n_frames = 4
    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._healthy_angle_range = healthy_angle_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=low, maxval=hi
    )
    qvel = jax.random.uniform(
        rng2, (self.sys.qd_size(),), minval=low, maxval=hi
    )

    pipeline_state = self.pipeline_init(qpos, qvel)

    obs = self._get_obs(pipeline_state)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_forward': zero,
        'reward_ctrl': zero,
        'reward_healthy': zero,
        'x_position': zero,
        'x_velocity': zero,
    }
    return State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    pipeline_state0 = state.pipeline_state
    pipeline_state = self.pipeline_step(pipeline_state0, action)
    obs = self._get_obs(pipeline_state)

    reward, rm = self.compute_reward(state.obs, obs, jp.zeros(()), action)

    done = self.is_done(obs)
    state.metrics.update(
        reward_forward=rm['forward_reward'],
        reward_ctrl=rm['ctrl_cost'],
        reward_healthy=rm['healthy_reward'],
        x_position=pipeline_state.x.pos[0, 0],
        x_velocity=rm['x_velocity'],
    )

    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

  def _get_obs(self, pipeline_state: base.State) -> jp.ndarray:
    """Returns the environment observations."""
    position = pipeline_state.q
    position = position.at[1].set(position[1] + 1.25) # height of the top
    # velocity = jp.clip(pipeline_state.qd, -10, 10)
    velocity = pipeline_state.qd

    if self._exclude_current_positions_from_observation:
      position = position[1:]

    return jp.concatenate((position, velocity))

  def is_done(self, next_obs: jp.ndarray) -> jp.ndarray:
    """Returns the done signal."""
    done = (1.0 - self._is_healthy(next_obs) if self._terminate_when_unhealthy
            else 0.0)
    return done

  def _is_healthy(self, next_obs: jp.ndarray) -> jp.ndarray:
    """Returns the healthy signal."""
    if self._exclude_current_positions_from_observation:
      z = next_obs[0]
      angle = next_obs[1]
    else:
      z = next_obs[1]
      angle = next_obs[2]

    min_z, max_z = self._healthy_z_range
    min_angle, max_angle = self._healthy_angle_range
    is_healthy = (
        (z > min_z) & (z < max_z) * (angle > min_angle) & (angle < max_angle)
    )
    return is_healthy

  def low_level_control(self, action: jp.ndarray,
                        obs: jp.ndarray) -> jp.ndarray:
    """There is no low level control, so return the action."""
    return action

  def dynamics_contact_integrate_only(
    self, obs: jp.ndarray, torques: jp.ndarray, ext_forces: jp.ndarray):

    q, qd = self.q_and_qd_from_obs(obs)

    sys = self.sys
    x, xd = kinematics.forward(sys, q, qd)
    state = GeneralizedState.init(q, qd, x, xd)
    state = dynamics.transform_com(sys, state)
    state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)
    state = state.replace(constraint_forces=jp.zeros_like(state.con_diag))
    tau = actuator.to_tau(sys, torques, state.q, state.qd)
    state = state.replace(qf_smooth=dynamics.forward(sys, state, tau))
    qf_constraint = ext_forces
    state = state.replace(qf_constraint=qf_constraint)

    def f(state, _):
        state = integrator.integrate(sys, state)
        return state, None

    state = jax.lax.scan(f, state, (), self._n_frames)[0]

    next_obs = self._get_obs(state)
    return next_obs

  def dynamics_contact(self, obs: jp.ndarray, torques: jp.ndarray,
                       ext_forces: jp.ndarray) -> jp.ndarray:
    # technically incorrect way to propagate the dynamics, but still gave good
    # results
    q, qd = self.q_and_qd_from_obs(obs)

    pipeline_state_start = self._pipeline_init_approx(q, qd)
    pipeline_state = self._pipeline_step_approx(pipeline_state_start,
                                                torques,
                                                ext_forces)

    next_obs = self._get_obs(pipeline_state)
    return next_obs

  def dynamics_mbpo(self, obs: jp.ndarray, torques_unused: jp.ndarray,
                    pred: jp.ndarray) -> jp.ndarray:
    next_obs = obs + pred
    return next_obs

  def dynamics_integrate(self, obs: jp.ndarray, torques_unused: jp.ndarray,
                         qdd: jp.ndarray) -> jp.ndarray:
    q, qd = self.q_and_qd_from_obs(obs)
    pipeline_state_start = self._pipeline_init_approx(q, qd)
    pipeline_state = integrator.integrate_qdd(self.sys, pipeline_state_start,
                                              qdd)
    next_obs = self._get_obs(pipeline_state)
    return next_obs

  def q_and_qd_from_obs(self, obs: jp.ndarray):
    if self._exclude_current_positions_from_observation:
      q = jp.zeros(self.sys.q_size())
      q = q.at[1:9].set(obs[:8])
      q = q.at[1].set(q[1] - 1.25)
      qd = obs[8:17]
    else:
      q = obs[:9]
      q = q.at[1].set(q[1] - 1.25)
      qd = obs[9:18]
    return q, qd

  def compute_reward(self, obs: jp.ndarray, obs_next: jp.ndarray,
                     unused_u: jp.ndarray, action: jp.ndarray):
    if self._exclude_current_positions_from_observation:
      x_velocity = obs_next[8]
    else:
      x_velocity = obs_next[9]
    forward_reward = self._forward_reward_weight * x_velocity

    is_healthy = self._is_healthy(obs_next)
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
    reward = forward_reward + healthy_reward - ctrl_cost

    reward_metrics = {
      'forward_reward': forward_reward,
      'ctrl_cost': -ctrl_cost,
      'healthy_reward': healthy_reward,
      'x_velocity': x_velocity
    }

    return reward, reward_metrics

  def ssrl_reward_fn(self, obs: jp.ndarray, obs_next: jp.ndarray,
                      unused_u: jp.ndarray, action: jp.ndarray):
    return self.compute_reward(obs, obs_next, unused_u, action)[0]

  def _pipeline_init_approx(self, q: jp.ndarray,
                          qd: jp.ndarray) -> base.State:
    """Initializes the pipeline state with the given q and qd"""
    sys = self.sys
    x, xd = kinematics.forward(sys, q, qd)
    state = GeneralizedState.init(q, qd, x, xd)  # pytype: disable=wrong-arg-types  # jax-ndarray
    state = dynamics.transform_com(sys, state)
    state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)
    state = state.replace(constraint_forces=jp.zeros_like(state.con_diag))

    return state

  def _pipeline_step_approx(
      self, pipeline_state: Any, act: jp.ndarray, ext_forces: jp.ndarray
  ) -> base.State:
    """Takes a physics step using the physics pipeline but with custom contact
    forces."""

    sys = self.sys

    def f(state, _):
        # calculate acceleration terms
        tau = actuator.to_tau(sys, act, state.q, state.qd)
        state = state.replace(qf_smooth=dynamics.forward(sys, state, tau))
        qf_constraint = ext_forces
        state = state.replace(qf_constraint=qf_constraint)

        # update position/velocity level terms
        state = integrator.integrate(sys, state)

        # we don't need to recompute x, xd since we are not using them to
        # propagate anything
        # x, xd = kinematics.forward(sys, state.q, state.qd)
        # state = state.replace(x=x, xd=xd)

        # these are disabled since they are not used to propagate the model
        # state
        # state = dynamics.transform_com(sys, state)
        # state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)
        # state = constraint.jacobian(sys, state, ignore_penetration=True)

        return state, None

    return jax.lax.scan(f, pipeline_state, (), self._n_frames)[0]
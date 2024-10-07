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
"""Trains a hopper to run in the +x direction."""

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


class Hopper2(PipelineEnv):



  # pyformat: disable
  """
  ### Description

  Same as Hopper, but with a separated out function for determining dones.

  This environment is based on the work done by Erez, Tassa, and Todorov in
  ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"](http://www.roboticsproceedings.org/rss07/p10.pdf). The environment aims to
  increase the number of independent state and control variables as compared to
  the classic control environments.

  The hopper is a two-dimensional one-legged figure that consist of four main
  body parts - the torso at the top, the thigh in the middle, the leg in the
  bottom, and a single foot on which the entire body rests.

  The goal is to make hops that move in the forward (right) direction by
  applying torques on the three hinges connecting the four body parts.

  ### Action Space

  The agent take a 3-element vector for actions. The action space is a
  continuous `(action, action, action)` all in `[-1, 1]`, where `action`
  represents the numerical torques applied between *links*

  | Num | Action                             | Control Min | Control Max | Name (in corresponding config) | Joint | Unit         |
  |-----|------------------------------------|-------------|-------------|--------------------------------|-------|--------------|
  | 0   | Torque applied on the thigh rotor  | -1          | 1           | thigh_joint                    | hinge | torque (N m) |
  | 1   | Torque applied on the leg rotor    | -1          | 1           | leg_joint                      | hinge | torque (N m) |
  | 3   | Torque applied on the foot rotor   | -1          | 1           | foot_joint                     | hinge | torque (N m) |

  ### Observation Space

  The state space consists of positional values of different body parts of the
  hopper, followed by the velocities of those individual parts (their
  derivatives) with all the positions ordered before all the velocities.

  The observation is a `ndarray` with shape `(11,)` where the elements
  correspond to the following:

  | Num | Observation                                      | Min  | Max | Name (in corresponding config) | Joint | Unit                     |
  |-----|--------------------------------------------------|------|-----|--------------------------------|-------|--------------------------|
  | 0   | z-coordinate of the top (height of hopper)       | -Inf | Inf | rootz                          | slide | position (m)             |
  | 1   | angle of the top                                 | -Inf | Inf | rooty                          | hinge | angle (rad)              |
  | 2   | angle of the thigh joint                         | -Inf | Inf | thigh_joint                    | hinge | angle (rad)              |
  | 3   | angle of the leg joint                           | -Inf | Inf | leg_joint                      | hinge | angle (rad)              |
  | 4   | angle of the foot joint                          | -Inf | Inf | foot_joint                     | hinge | angle (rad)              |
  | 5   | velocity of the x-coordinate of the top          | -Inf | Inf | rootx                          | slide | velocity (m/s)           |
  | 6   | velocity of the z-coordinate (height) of the top | -Inf | Inf | rootz                          | slide | velocity (m/s)           |
  | 7   | angular velocity of the angle of the top         | -Inf | Inf | rooty                          | hinge | angular velocity (rad/s) |
  | 8   | angular velocity of the thigh hinge              | -Inf | Inf | thigh_joint                    | hinge | angular velocity (rad/s) |
  | 9   | angular velocity of the leg hinge                | -Inf | Inf | leg_joint                      | hinge | angular velocity (rad/s) |
  | 10  | angular velocity of the foot hinge               | -Inf | Inf | foot_joint                     | hinge | angular velocity (rad/s) |

  ### Rewards

  The reward consists of three parts:

  - *reward_healthy*: Every timestep that the hopper is alive, it gets a reward
    of 1,
  - *reward_forward*: A reward of hopping forward which is measured as
    *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the
    time between actions - the default *dt = 0.008*. This reward would be
    positive if the hopper hops forward (right) desired.
  - *reward_ctrl*: A negative reward for penalising the hopper if it takes
    actions that are too large. It is measured as *-coefficient **x**
    sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
    control and has a default value of 0.001

  ### Starting State

  All observations start in state (0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0) with a uniform noise in the range of [-0.005, 0.005] added to the
  values for stochasticity.

  ### Episode Termination

  The episode terminates when any of the following happens:

  1. The episode duration reaches a 1000 timesteps
  2. The height of the hopper becomes less than 0.7 metres (hopper has hopped
     too low).
  3. The absolute value of the angle (index 2) is less than 0.2 radians (hopper
     has fallen down).

  ### Arguments

  No additional arguments are currently supported (in v2 and lower), but
  modifications can be made to the XML file in the assets folder
  (or by changing the path to a modified XML file in another folder).

  ```
  env = gym.make('Hopper-v2')
  ```

  v3, v4, and v5 take gym.make kwargs such as ctrl_cost_weight,
  reset_noise_scale etc.

  ```
  env = gym.make('Hopper-v5', ctrl_cost_weight=0.1, ....)
  ```

  ### Version History

  * v5: ported to Brax.
  * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
  * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight,
        reset_noise_scale etc. rgb rendering comes from tracking camera (so
        agent does not run away from screen)
  * v2: All continuous control environments now use mujoco_py >= 1.50
  * v1: max_time_steps raised to 1000 for robot based tasks. Added
        reward_threshold to environments.
  * v0: Initial versions release (1.0.0)
  """
  # pyformat: enable


  def __init__(
      self,
      forward_reward_weight: float = 1.0,
      ctrl_cost_weight: float = 1e-3,
      healthy_reward: float = 1.0,
      terminate_when_unhealthy: bool = True,
      healthy_state_range=(-100.0, 100.0),
      healthy_z_range: Tuple[float, float] = (0.7, float('inf')),
      healthy_angle_range=(-0.2, 0.2),
      reset_noise_scale=5e-3,
      exclude_current_positions_from_observation=True,
      backend='generalized',
      forces_in_q_coords=True,
      **kwargs
  ):
    """Creates a Hopper environment.

    Args:
      forward_reward_weight: Weight for the forward reward, i.e. velocity in
        x-direction.
      ctrl_cost_weight: Weight for the control cost.
      healthy_reward: Reward for staying healthy, i.e. respecting the posture
        constraints.
      terminate_when_unhealthy: Done bit will be set when unhealthy if true.
      healthy_state_range: state range for the hopper to be considered healthy.
      healthy_z_range: Range of the z-position for being healthy.
      healthy_angle_range: Range of joint angles for being healthy.
      reset_noise_scale: Scale of noise to add to reset states.
      exclude_current_positions_from_observation: x-position will be hidden from
        the observations if true.
      backend: str, the physics backend to use
      **kwargs: Arguments that are passed to the base class.
    """
    path = epath.resource_path('brax') / 'envs/assets/hopper.xml'
    sys = mjcf.load(path)
    sys = sys.replace(matrix_inv_iterations=0)

    n_frames = 4
    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_state_range = healthy_state_range
    self._healthy_z_range = healthy_z_range
    self._healthy_angle_range = healthy_angle_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

    if forces_in_q_coords:
      self._qfc_fn = lambda state, forces: forces
    else:
      # self._qfc_fn = lambda state, forces: state.con_jac.T @ forces
      raise NotImplementedError('forces_in_q_coords=False not supported'
                                ' (constraint jacobian is not being computed'
                                'in dynamics fn)')

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
      state_vec = next_obs[1:]
      z = next_obs[0]
      angle = next_obs[1]
    else:
      state_vec = next_obs[2:]
      z = next_obs[1]
      angle = next_obs[2]

    min_z, max_z = self._healthy_z_range
    min_angle, max_angle = self._healthy_angle_range
    min_state, max_state = self._healthy_state_range
    is_healthy = jp.all(
        jp.logical_and(min_state < state_vec, state_vec < max_state)
    )
    is_healthy &= jp.logical_and(min_z < z, z < max_z)
    is_healthy &= jp.logical_and(min_angle < angle, angle < max_angle)
    return is_healthy

  def low_level_control(self, action: jp.ndarray,
                        obs: jp.ndarray) -> jp.ndarray:
    """There is no low level control, so return the action."""
    return action

  def dynamics_contact_correct(self, obs: jp.ndarray, torques: jp.ndarray,
                               ext_forces: jp.ndarray) -> jp.ndarray:
    # initialize state
    sys = self.sys
    q, qd = self.q_and_qd_from_obs(obs)
    x, xd = kinematics.forward(sys, q, qd)
    state = GeneralizedState.init(q, qd, x, xd)
    state = dynamics.transform_com(sys, state)
    state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)
    state = state.replace(constraint_forces=jp.zeros_like(state.con_diag))

    # propagate dynamics
    def f(state, idx):
      tau = actuator.to_tau(sys, torques, state.q, state.qd)
      state = state.replace(qf_smooth=dynamics.forward(sys, state, tau))
      state = state.replace(qf_constraint=ext_forces)
      state = integrator.integrate(sys, state)

      def update_internal_brax_states(state):
        x, xd = kinematics.forward(sys, state.q, state.qd)
        state = state.replace(x=x, xd=xd)
        state = dynamics.transform_com(sys, state)
        state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)
        return state

      # don't need to update interal brax states for the last n_frame
      state = jax.lax.cond(
          idx < self._n_frames - 1,
          update_internal_brax_states,
          lambda x: x,
          state,
      )

      return state, None

    nstate = jax.lax.scan(f, state, jp.arange(self._n_frames))[0]
    next_obs = self._get_obs(nstate)
    return next_obs

  def dynamics_contact_zoh(self, obs: jp.ndarray, torques: jp.ndarray,
                           ext_forces: jp.ndarray) -> jp.ndarray:
    # here we are performing only one integration step with a dt of
    # (self.sys.dt * self._n_frames). We assume zero order hold for both the
    # torques and the ext_forces

    q, qd = self.q_and_qd_from_obs(obs)

    sys = self.sys.replace(dt=self.dt)  # self.sys.dt * self._n_frames
    x, xd = kinematics.forward(sys, q, qd)
    state = GeneralizedState.init(q, qd, x, xd)
    state = dynamics.transform_com(sys, state)
    state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)
    state = state.replace(constraint_forces=jp.zeros_like(state.con_diag))

    tau = actuator.to_tau(sys, torques, state.q, state.qd)
    state = state.replace(qf_smooth=dynamics.forward(sys, state, tau))
    qf_constraint = self._qfc_fn(state, ext_forces)
    state = state.replace(qf_constraint=qf_constraint)
    state = integrator.integrate(sys, state)

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
    qf_constraint = self._qfc_fn(state, ext_forces)
    state = state.replace(qf_constraint=qf_constraint)

    def f(state, _):
        state = integrator.integrate(sys, state)
        return state, None

    state = jax.lax.scan(f, state, (), self._n_frames)[0]

    next_obs = self._get_obs(state)
    return next_obs

  def dynamics_all_forces(self, obs: jp.ndarray, torques: jp.ndarray,
                          ext_forces: jp.ndarray) -> jp.ndarray:
    q, qd = self.q_and_qd_from_obs(obs)

    sys = self.sys
    x, xd = kinematics.forward(sys, q, qd)
    state = GeneralizedState.init(q, qd, x, xd)
    state = dynamics.transform_com(sys, state)
    state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)
    state = state.replace(constraint_forces=jp.zeros_like(state.con_diag))

    def f(state, _):
        # calculate acceleration terms
        tau = actuator.to_tau(sys, torques, state.q, state.qd)
        state = state.replace(qf_smooth=tau)  # only use the torques
        qf_constraint = self._qfc_fn(state, ext_forces)
        state = state.replace(qf_constraint=qf_constraint)

        # update position/velocity level terms
        state = integrator.integrate(sys, state)

        return state, None

    state = jax.lax.scan(f, state, (), self._n_frames)[0]

    next_obs = self._get_obs(state)
    return next_obs

  def dynamics_all_forces_correct(self, obs: jp.ndarray, torques: jp.ndarray,
                               ext_forces: jp.ndarray) -> jp.ndarray:
    # initialize state
    sys = self.sys
    q, qd = self.q_and_qd_from_obs(obs)
    x, xd = kinematics.forward(sys, q, qd)
    state = GeneralizedState.init(q, qd, x, xd)
    state = dynamics.transform_com(sys, state)
    state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)
    state = state.replace(constraint_forces=jp.zeros_like(state.con_diag))

    # propagate dynamics
    def f(state, idx):
      tau = actuator.to_tau(sys, torques, state.q, state.qd)
      state = state.replace(qf_smooth=tau)  # only use the torques
      state = state.replace(qf_constraint=ext_forces)
      state = integrator.integrate(sys, state)

      def update_internal_brax_states(state):
        x, xd = kinematics.forward(sys, state.q, state.qd)
        state = state.replace(x=x, xd=xd)
        state = dynamics.transform_com(sys, state)
        state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)
        return state

      # don't need to update interal brax states for the last n_frame
      state = jax.lax.cond(
          idx < self._n_frames - 1,
          update_internal_brax_states,
          lambda x: x,
          state,
      )

      return state, None

    nstate = jax.lax.scan(f, state, jp.arange(self._n_frames))[0]
    next_obs = self._get_obs(nstate)
    return next_obs

  def dynamics_integrate(self, obs: jp.ndarray, torques_unused: jp.ndarray,
                         qdd: jp.ndarray) -> jp.ndarray:
    q, qd = self.q_and_qd_from_obs(obs)
    pipeline_state_start = self._pipeline_init_approx(q, qd)
    pipeline_state = integrator.integrate_qdd(self.sys, pipeline_state_start,
                                              qdd)
    next_obs = self._get_obs(pipeline_state)
    return next_obs

  def dynamics_mbpo(self, obs: jp.ndarray, torques_unused: jp.ndarray,
                    pred: jp.ndarray) -> jp.ndarray:
    next_obs = obs + pred
    return next_obs

  def dynamics_exact(self, obs: jp.ndarray, torques: jp.ndarray,
                     pred_unused: jp.ndarray) -> jp.ndarray:
    q, qd = self.q_and_qd_from_obs(obs)

    pipeline_state_start = self._pipeline_init_exact(q, qd)
    pipeline_state = self.pipeline_step(pipeline_state_start, torques)

    next_obs = self._get_obs(pipeline_state)
    return next_obs

  def q_and_qd_from_obs(self, obs: jp.ndarray):
    if self._exclude_current_positions_from_observation:
      q = jp.zeros(self.sys.q_size())
      q = q.at[1:6].set(obs[:5])
      q = q.at[1].set(q[1] - 1.25)
      qd = obs[5:11]
    else:
      q = obs[:6]
      q = q.at[1].set(q[1] - 1.25)
      qd = obs[6:12]
    return q, qd

  def compute_reward(self, obs: jp.ndarray, obs_next: jp.ndarray,
                     unused_u: jp.ndarray, action: jp.ndarray):
    if self._exclude_current_positions_from_observation:
      x_velocity = obs_next[5]
    else:
      x_velocity = obs_next[6]
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


  def _pipeline_init_exact(self, q: jp.ndarray,
                          qd: jp.ndarray) -> base.State:
    """Initializes the pipeline state with the given q and qd"""
    return self._pipeline.init(self.sys, q, qd, self._debug)

  def _pipeline_init_approx(self, q: jp.ndarray,
                          qd: jp.ndarray) -> base.State:
    """Initializes the pipeline state with the given q and qd"""
    sys = self.sys
    x, xd = kinematics.forward(sys, q, qd)
    state = GeneralizedState.init(q, qd, x, xd)
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
        qf_constraint = self._qfc_fn(state, ext_forces)
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

from brax.robots.go1.utils import Go1Utils
from brax.robots.go1.gait import Go1Gait, Go1GaitParams
from brax.envs.base import RlwamEnv, State

from brax import actuator
from brax import kinematics
from brax.generalized.base import State as GeneralizedState
from brax.generalized import dynamics
from brax.generalized import integrator
from brax.generalized import mass
from brax import base
from brax.math import rotate, inv_rotate, quat_to_eulerzyx, eulerzyx_to_quat
from brax.generalized.pipeline import step as pipeline_step

from jax import numpy as jp
from typing import Optional, Any, Tuple, Callable
import jax
import flax


@flax.struct.dataclass
class ControlCommand:
    """Output of the low level controller which includes gait control and
    inverse kinematics. """
    q_des: jp.ndarray
    qd_des: jp.ndarray
    Kp: jp.ndarray
    Kd: jp.ndarray
    contact: jp.ndarray
    leg_phases: jp.ndarray
    pdes: jp.ndarray


class Go1GoFast(RlwamEnv):
    """ Go1 environment

    ### Observation space

    | Num | Observation                          | Min  | Max | Name           | Joint | Unit                 |
    | --- | ------------------------------------ | ---- | --- | -------------- | ----- | -------------------- |
    | 0   | w-quat of trunk                      | -1   | 1   | rw             | free  | angle (quat)         |
    | 1   | x-quat of trunk                      | -1   | 1   | rw             | free  | angle (quat)         |
    | 2   | y-quat of trunk                      | -1   | 1   | rw             | free  | angle (quat)         |
    | 3   | z-quat of trunk                      | -1   | 1   | rw             | free  | angle (quat)         |
    | 4   | front-right hip joint angle          | -Inf | Inf | FR_hip_joint   | hinge | angle (rad)          |
    | 5   | front-right thigh joint angle        | -Inf | Inf | FR_thigh_joint | hinge | angle (rad)          |
    | 6   | front-right calf joint angle         | -Inf | Inf | FR_calf_joint  | hinge | angle (rad)          |
    | 7   | front-left hip joint angle           | -Inf | Inf | FL_hip_joint   | hinge | angle (rad)          |
    | 8   | front-left thigh joint angle         | -Inf | Inf | FL_thigh_joint | hinge | angle (rad)          |
    | 9   | front-left calf joint angle          | -Inf | Inf | FL_calf_joint  | hinge | angle (rad)          |
    | 10  | rear-right hip joint angle           | -Inf | Inf | RR_hip_joint   | hinge | angle (rad)          |
    | 11  | rear-right thigh joint angle         | -Inf | Inf | RR_thigh_joint | hinge | angle (rad)          |
    | 12  | rear-right calf joint angle          | -Inf | Inf | RR_calf_joint  | hinge | angle (rad)          |
    | 13  | rear-left hip joint angle            | -Inf | Inf | RL_hip_joint   | hinge | angle (rad)          |
    | 14  | rear-left thigh joint angle          | -Inf | Inf | RL_thigh_joint | hinge | angle (rad)          |
    | 15  | rear-left calf joint angle           | -Inf | Inf | RL_calf_joint  | hinge | angle (rad)          |
    | 16  | x-velocity of trunk (body frame)     | -Inf | Inf | vx             | free  | velocity (m/s)       |
    | 17  | y-velocity of trunk (body frame)     | -Inf | Inf | vy             | free  | velocity (m/s)       |
    | 18  | z-velocity of trunk (body frame)     | -Inf | Inf | vz             | free  | velocity (m/s)       |
    | 19  | x-ang-velocity of trunk (body frame) | -Inf | Inf | wx             | free  | ang-velocity (rad/s) |
    | 20  | y-ang-velocity of trunk (body frame) | -Inf | Inf | wy             | free  | ang-velocity (rad/s) |
    | 21  | z-ang-velocity of trunk (body frame) | -Inf | Inf | wz             | free  | ang-velocity (rad/s) |
    | 22  | front-right hip joint speed          | -Inf | Inf | FR_hip_speed   | hinge | ang-speed (rad/s)    |
    | 23  | front-right thigh joint speed        | -Inf | Inf | FR_thigh_speed | hinge | ang-speed (rad/s)    |
    | 24  | front-right calf joint speed         | -Inf | Inf | FR_calf_speed  | hinge | ang-speed (rad/s)    |
    | 25  | front-left hip joint speed           | -Inf | Inf | FL_hip_speed   | hinge | ang-speed (rad/s)    |
    | 26  | front-left thigh joint speed         | -Inf | Inf | FL_thigh_speed | hinge | ang-speed (rad/s)    |
    | 27  | front-left calf joint speed          | -Inf | Inf | FL_calf_speed  | hinge | ang-speed (rad/s)    |
    | 28  | rear-right hip joint speed           | -Inf | Inf | RR_hip_speed   | hinge | ang-speed (rad/s)    |
    | 29  | rear-right thigh joint speed         | -Inf | Inf | RR_thigh_speed | hinge | ang-speed (rad/s)    |
    | 30  | rear-right calf joint speed          | -Inf | Inf | RR_calf_speed  | hinge | ang-speed (rad/s)    |
    | 31  | rear-left hip joint speed            | -Inf | Inf | RL_hip_speed   | hinge | ang-speed (rad/s)    |
    | 32  | rear-left thigh joint speed          | -Inf | Inf | RL_thigh_speed | hinge | ang-speed (rad/s)    |
    | 33  | rear-left calf joint speed           | -Inf | Inf | RL_calf_speed  | hinge | ang-speed (rad/s)    |
    | 34  | cos(phase)                           | -1   | 1   | cos_phase      | none  | unitless             |
    | 35  | sin(phase)                           | -1   | 1   | sin_phase      | none  | unitless             |

    ### Action space
    | Num   | Action                           | Min | Max |
    | ----- | -------------------------------- | --- | --- |
    | 0:4   | x foot position deltas           |     |     |
    | 4:8   | y foot position deltas           |     |     |
    | 8     | Body height, delta from standing |     |     |
    | 9:21  | P gains (if enabled)             |     |     |
    | 21:33 | D gains (if enabled)             |     |     |


    ### Actuator space

    | Num | Actuator                       | Min  | Max | Name     | Joint | Unit         |
    | --- | ------------------------------ | ---- | --- | -------- | ----- | ------------ |
    | 0   | front-right hip joint torque   | -Inf | Inf | FR_hip   | hinge | torque (N*m) |
    | 1   | front-right thigh joint torque | -Inf | Inf | FR_thigh | hinge | torque (N*m) |
    | 2   | front-right calf joint torque  | -Inf | Inf | FR_calf  | hinge | torque (N*m) |
    | 3   | front-left hip joint torque    | -Inf | Inf | FL_hip   | hinge | torque (N*m) |
    | 4   | front-left thigh joint torque  | -Inf | Inf | FL_thigh | hinge | torque (N*m) |
    | 5   | front-left calf joint torque   | -Inf | Inf | FL_calf  | hinge | torque (N*m) |
    | 6   | rear-right hip joint torque    | -Inf | Inf | RR_hip   | hinge | torque (N*m) |
    | 7   | rear-right thigh joint torque  | -Inf | Inf | RR_thigh | hinge | torque (N*m) |
    | 8   | rear-right calf joint torque   | -Inf | Inf | RR_calf  | hinge | torque (N*m) |
    | 9   | rear-left hip joint torque     | -Inf | Inf | RL_hip   | hinge | torque (N*m) |
    | 10  | rear-left thigh joint torque   | -Inf | Inf | RL_thigh | hinge | torque (N*m) |
    | 11  | rear-left calf joint torque    | -Inf | Inf | RL_calf  | hinge | torque (N*m) |

    Disable flake8 line-too-long errors: # noqa: E501
    """

    def __init__(
        self,
        policy_repeat=4,
        forward_cmd_vel_type='constant',  # 'constant' or 'sine'
        forward_cmd_vel_range=(0.0, 0.0),  # for now just using the average of this for the gait controller
        forward_cmd_vel_period_range=(5.0, 10.0),  # only used with 'sine'
        turn_cmd_rate_range=(-jp.pi/8, jp.pi/8),
        initial_yaw_range=(-0.0, 0.0),
        contact_time_const=0.02,
        contact_time_const_range=None,
        contact_damping_ratio=1.0,
        friction_range=(0.6, 0.6),
        ground_roll_range=(0.0, 0.0),
        ground_pitch_range=(0.0, 0.0),
        joint_damping_perc_range=(1.0, 1.0),
        joint_gain_range=(1.0, 1.0),
        link_mass_perc_range=(1.0, 1.0),
        forward_vel_rew_weight=1.0,
        turn_rew_weight=0.5,
        pitch_rew_weight=0.20,
        roll_rew_weight=0.25,
        yaw_rew_weight=0.00,
        side_motion_rew_weight=0.25,
        z_vel_change_rew_weight=0.0,
        ang_vel_rew_weight=0.00,
        ang_change_rew_weight=0.25,
        joint_lim_rew_weight=0.15,
        torque_lim_rew_weight=0.15,
        joint_acc_rew_weight=0.05,
        action_rew_weight=0.1,
        cosmetic_rew_weight=0.0,
        energy_rew_weight=0.0,
        foot_z_rew_weight=0.0,
        torque_lim_penalty_weight=1.0,
        fallen_roll=jp.pi/4,
        fallen_pitch=jp.pi/4,
        forces_in_q_coords=False,
        include_height_in_obs=False,
        body_height_in_action_space=True,
        gains_in_action_space=False,
        backend='generalized',
        reward_type='normalized',
        used_cached_systems=False,
        healthy_delta_radius=2.0,  # not used
        healthy_delta_yaw=1.57,  # not used
        **kwargs
    ):

        self.sim_dt = 1/400  # simulation dt; 400 Hz

        # determines high level policy freq; (1/sim_dt)/policy_repeat Hz
        self.policy_repeat = policy_repeat

        sys = Go1Utils.get_system(used_cached_systems)
        sys = sys.replace(dt=self.sim_dt)

        # here we are using the fast sim_dt with the approximate system instead
        # of self.sim_dt * self.policy_repeat
        self._sys_approx = Go1Utils.get_approx_system(used_cached_systems)
        self._sys_approx = self._sys_approx.replace(dt=self.sim_dt)

        # normally this is use by Brax as the number of times to step the
        # physics pipeline for each environment step. However we have
        # overwritten the pipline_step function with our own behaviour which
        # steps the physics self.policy_repeat times. So we set this to 1.
        n_frames = 1
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._period = 0.50  # period of the gait cycle (sec)
        self._forward_cmd_vel = jp.mean(jp.array(forward_cmd_vel_range))
        self._initial_yaw_range = initial_yaw_range
        if contact_time_const_range is None:
            self._contact_time_const_range = (contact_time_const,
                                              contact_time_const)
        else:
            self._contact_time_const_range = contact_time_const_range
        self._contact_damping_ratio = contact_damping_ratio
        self._friction_range = friction_range
        self._ground_roll_range = ground_roll_range
        self._ground_pitch_range = ground_pitch_range
        self._joint_damping_perc_range = joint_damping_perc_range
        self._joint_gain_range = joint_gain_range
        self._link_mass_perc_range = link_mass_perc_range
        self._fallen_roll = fallen_roll
        self._fallen_pitch = fallen_pitch
        self._include_height_in_obs = include_height_in_obs
        self._body_height_in_action_space = body_height_in_action_space
        self._gains_in_action_space = gains_in_action_space

        if reward_type == 'normalized':
            self._reward_fn = self._reward_normalized
        elif reward_type == 'weighted_quadratic':
            self._reward_fn = self._reward_weighted_quadratic
        elif reward_type == 'quadratic_hardware':
            self._reward_fn = self._reward_quadratic_hardware
        else:
            raise ValueError(f'Invalid reward_type: {reward_type}')

        if forces_in_q_coords:
            self._qfc_fn = lambda state, forces: forces
        else:
            self._qfc_fn = lambda state, forces: state.con_jac.T @ forces

        # set up slices for the state space, defined in the xml file
        self._xml_quat_idxs = jp.s_[3:7]
        self._xml_q_idxs = jp.s_[7:19]
        self._xml_base_vel_idxs = jp.s_[0:3]
        self._xml_rpy_rate_idxs = jp.s_[3:6]
        self._xml_qd_idxs = jp.s_[6:18]
        self._xml_h_idxs = jp.s_[2:3]

        # set up slices for the observation space, defined by the table above
        self._quat_idxs = jp.s_[0:4]
        self._q_idxs = jp.s_[4:16]
        self._base_vel_idxs = jp.s_[16:19]
        self._forward_vel_idx = jp.s_[16]
        self._y_vel_idx = jp.s_[17]
        self._z_vel_idx = jp.s_[18]
        self._rpy_rate_idxs = jp.s_[19:22]
        self._roll_rate_idx = jp.s_[19]
        self._pitch_rate_idx = jp.s_[20]
        self._turn_rate_idx = jp.s_[21]
        self._qd_idxs = jp.s_[22:34]
        self._cos_phase_idx = jp.s_[34]
        self._sin_phase_idx = jp.s_[35]
        self._h_idx = jp.s_[36]

        # set up observation normalization limits
        self._observation_size = 37 if self._include_height_in_obs else 36
        self.obs_limits = jp.ones((self.observation_size, 2))
        self.obs_limits = self.obs_limits.at[:, 0].set(-1.)
        self.obs_limits = self.obs_limits.at[self._q_idxs, 0].set(
            jp.tile(Go1Utils.LOWER_JOINT_LIMITS, 4) - 0.25)
        self.obs_limits = self.obs_limits.at[self._q_idxs, 1].set(
            jp.tile(Go1Utils.UPPER_JOINT_LIMITS, 4) + 0.25)
        self.obs_limits = self.obs_limits.at[self._forward_vel_idx, 0].set(-0.2)
        self.obs_limits = self.obs_limits.at[self._forward_vel_idx, 1].set(2.5)
        self.obs_limits = self.obs_limits.at[self._y_vel_idx, 0].set(-0.5)
        self.obs_limits = self.obs_limits.at[self._y_vel_idx, 1].set(0.5)
        self.obs_limits = self.obs_limits.at[self._z_vel_idx, 0].set(-0.25)
        self.obs_limits = self.obs_limits.at[self._z_vel_idx, 1].set(0.25)
        self.obs_limits = self.obs_limits.at[self._roll_rate_idx, 0].set(-1.5)
        self.obs_limits = self.obs_limits.at[self._roll_rate_idx, 1].set(1.5)
        self.obs_limits = self.obs_limits.at[self._pitch_rate_idx, 0].set(-1.5)
        self.obs_limits = self.obs_limits.at[self._pitch_rate_idx, 1].set(1.5)
        self.obs_limits = self.obs_limits.at[self._turn_rate_idx, 0].set(-1.5)
        self.obs_limits = self.obs_limits.at[self._turn_rate_idx, 1].set(1.5)
        self.obs_limits = self.obs_limits.at[self._qd_idxs, 0].set(-3.5)
        self.obs_limits = self.obs_limits.at[self._qd_idxs, 1].set(3.5)
        self.obs_limits = self.obs_limits.at[self._cos_phase_idx, 0].set(-1.0)
        self.obs_limits = self.obs_limits.at[self._cos_phase_idx, 1].set(1.0)
        self.obs_limits = self.obs_limits.at[self._sin_phase_idx, 0].set(-1.0)
        self.obs_limits = self.obs_limits.at[self._sin_phase_idx, 1].set(1.0)
        if self._include_height_in_obs:
            self.obs_limits = self.obs_limits.at[self._h_idx, 0].set(0.0)
            self.obs_limits = self.obs_limits.at[self._h_idx, 1].set(0.5)

        # set up slices for the action space, defined by the table above
        self._action_size = 8
        if self._body_height_in_action_space:
            self._action_size = 33 if self._gains_in_action_space else 9
        self._ac_delta_pdes_idxs = jp.s_[0:8]
        self._ac_delta_pdes_x_idxs = jp.s_[0:4]
        self._ac_delta_pdes_y_idxs = jp.s_[4:8]
        self._ac_dbody_h_idx = jp.s_[8]
        self._ac_Kp_idxs = jp.s_[9:21]
        self._ac_Kd_idxs = jp.s_[21:33]

        # define action space
        dx = 0.150
        dy = 0.075
        dKp = 30.0
        dKd = 1.0
        self._ac_space = jp.ones((self.action_size, 2))
        self._ac_space = self._ac_space.at[:, 0].set(-1.)
        self._ac_space = self._ac_space.at[self._ac_delta_pdes_x_idxs, 0].set(-dx)
        self._ac_space = self._ac_space.at[self._ac_delta_pdes_x_idxs, 1].set(dx)
        self._ac_space = self._ac_space.at[self._ac_delta_pdes_y_idxs, 0].set(-dy)
        self._ac_space = self._ac_space.at[self._ac_delta_pdes_y_idxs, 1].set(dy)
        if self._body_height_in_action_space:
            self._ac_space = self._ac_space.at[self._ac_dbody_h_idx, 0].set(-0.1)
            self._ac_space = self._ac_space.at[self._ac_dbody_h_idx, 1].set(0.0)
        if self._gains_in_action_space:
            self._ac_space = self._ac_space.at[self._ac_Kp_idxs, 0].set(-dKp)
            self._ac_space = self._ac_space.at[self._ac_Kp_idxs, 1].set(dKp)
            self._ac_space = self._ac_space.at[self._ac_Kd_idxs, 0].set(-dKd)
            self._ac_space = self._ac_space.at[self._ac_Kd_idxs, 1].set(dKd)

        # set up reward weights whose sum is 1
        self._reward_weights = jp.array([
            forward_vel_rew_weight,
            turn_rew_weight,
            pitch_rew_weight,
            roll_rew_weight,
            yaw_rew_weight,
            side_motion_rew_weight,
            z_vel_change_rew_weight,
            ang_vel_rew_weight,
            ang_change_rew_weight,
            joint_lim_rew_weight,
            torque_lim_rew_weight,
            joint_acc_rew_weight,
            action_rew_weight,
            cosmetic_rew_weight,
            energy_rew_weight,
            foot_z_rew_weight
        ])
        self._reward_weights = self._reward_weights / self._reward_weights.sum()
        # add torque penalty weight; this is not included in the normalization
        self._reward_weights = jp.concatenate(
            [self._reward_weights, jp.array([torque_lim_penalty_weight])]
        )

    def reset(self, rng: jp.ndarray) -> State:

        # randomize initial yaw
        rng, rng_yaw = jax.random.split(rng)
        initial_yaw = jax.random.uniform(
            rng_yaw, shape=(),
            minval=self._initial_yaw_range[0]*180/jp.pi,
            maxval=self._initial_yaw_range[1]*180/jp.pi
        )
        initial_quat = eulerzyx_to_quat(jp.array([0.0, 0.0, initial_yaw]))

        # initialize system with initial q and qd
        q = self.sys.init_q  # init_q is defined in the xml file
        q = q.at[self._xml_quat_idxs].set(initial_quat)
        qd = jp.zeros(self.sys.qd_size())
        pipeline_state = self.pipeline_init(q, qd)

        # domain randomization
        domain_rand_rngs = jax.random.split(rng, 7)
        self._contact_time_const = jax.random.uniform(
            domain_rand_rngs[0], shape=(),
            minval=self._contact_time_const_range[0],
            maxval=self._contact_time_const_range[1]
        )
        self._friction = jax.random.uniform(
            domain_rand_rngs[1], shape=(),
            minval=self._friction_range[0],
            maxval=self._friction_range[1]
        )
        self._ground_roll = jax.random.uniform(
            domain_rand_rngs[2], shape=(),
            minval=self._ground_roll_range[0],
            maxval=self._ground_roll_range[1]
        )
        self._ground_pitch = jax.random.uniform(
            domain_rand_rngs[3], shape=(),
            minval=self._ground_pitch_range[0],
            maxval=self._ground_pitch_range[1]
        )
        self._joint_damping = jax.random.uniform(
            domain_rand_rngs[4], shape=self.sys.dof.damping.shape,
            minval=self._joint_damping_perc_range[0] * self.sys.dof.damping,
            maxval=self._joint_damping_perc_range[1] * self.sys.dof.damping
        )
        self._joint_gain = jax.random.uniform(
            domain_rand_rngs[5], shape=self.sys.actuator.gain.shape,
            minval=self._joint_gain_range[0],
            maxval=self._joint_gain_range[1]
        )
        self._link_mass_perc = jax.random.uniform(
            domain_rand_rngs[6], shape=self.sys.link.inertia.mass.shape,
            minval=self._link_mass_perc_range[0],
            maxval=self._link_mass_perc_range[1]
        )

        # initialize metrics
        metrics = {
            'step_count': jp.zeros(()),  # used for phase
            'forward_vel': jp.zeros(()),
            'rew_forward_vel': jp.zeros(()),
            'rew_turn': jp.zeros(()),
            'rew_pitch': jp.zeros(()),
            'rew_roll': jp.zeros(()),
            'rew_yaw': jp.zeros(()),
            'rew_side_motion': jp.zeros(()),
            'rew_z_vel_change': jp.zeros(()),
            'rew_ang_vel': jp.zeros(()),
            'rew_ang_change': jp.zeros(()),
            'rew_joint_limits': jp.zeros(()),
            'rew_torque_limits': jp.zeros(()),
            'rew_joint_acc': jp.zeros(()),
            'rew_action': jp.zeros(()),
            'rew_cosmetic': jp.zeros(()),
            'rew_energy': jp.zeros(()),
            'rew_foot_z': jp.zeros(()),
            'penalty_torque_lim': jp.zeros(()),
        }

        # we use info to pass along quantities for domain randomization
        info = {
            'contact_time_const': self._contact_time_const,
            'contact_damping_ratio': self._contact_damping_ratio,
            'friction': self._friction,
            'ground_roll': self._ground_roll,
            'ground_pitch': self._ground_pitch,
            'joint_damping': self._joint_damping,
            'joint_gain': self._joint_gain,
            'link_mass_perc': self._link_mass_perc,
        }

        empty_cmd = ControlCommand(
            q_des=jp.zeros((12,)),
            qd_des=jp.zeros((12,)),
            Kp=jp.zeros((12,)),
            Kd=jp.zeros((12,)),
            contact=jp.array([True, True, True, True]),
            leg_phases=jp.zeros((4,)),
            pdes=jp.zeros((12,))
        )
        info['cmd'] = empty_cmd

        # initial observations, reward, done, and u
        norm_obs = self._get_obs(pipeline_state, metrics)
        reward, done = jp.zeros(2)
        u = jp.zeros(self._action_size)

        return State(pipeline_state, norm_obs, reward, done, metrics,
                     info=info, u=u)

    def step(self, state: State, action: jp.ndarray) -> State:

        # overwrite system contact properties with the environment's
        sys = self._update_system_properties(state)

        # get observations from state
        prev_norm_obs = self._get_obs(state.pipeline_state, state.metrics)

        # scale action
        scaled_action = self.scale_action(action)

        def f(pipeline_state, _):
            norm_obs = self._get_obs(pipeline_state, state.metrics)
            u, _ = self.torque_pd_control(scaled_action, norm_obs)
            pipeline_state = pipeline_step(sys, pipeline_state, u)
            return pipeline_state, _

        new_pipeline_state, _ = jax.lax.scan(f, state.pipeline_state,
                                             (), self.policy_repeat)

        # get new observations and compute reward
        state.metrics.update(
            step_count=state.metrics['step_count'] + 1
        )
        new_norm_obs = self._get_obs(new_pipeline_state, state.metrics)
        state.metrics.update(
            forward_vel=self._denormalize_obs(new_norm_obs)[self._forward_vel_idx]
        )

        reward, rew_components = self.compute_reward(
            new_norm_obs, prev_norm_obs, jp.zeros(()), action)

        state.metrics.update(rew_forward_vel=rew_components['rew_forward_vel'],
                             rew_turn=rew_components['rew_turn'],
                             rew_pitch=rew_components['rew_pitch'],
                             rew_roll=rew_components['rew_roll'],
                             rew_yaw=rew_components['rew_yaw'],
                             rew_side_motion=rew_components['rew_side_motion'],
                             rew_z_vel_change=rew_components['rew_z_vel_change'],
                             rew_ang_vel=rew_components['rew_ang_vel'],
                             rew_ang_change=rew_components['rew_ang_change'],
                             rew_joint_limits=rew_components['rew_joint_limits'],
                             rew_torque_limits=rew_components['rew_torque_limits'],
                             rew_joint_acc=rew_components['rew_joint_acc'],
                             rew_action=rew_components['rew_action'],
                             rew_cosmetic=rew_components['rew_cosmetic'],
                             rew_energy=rew_components['rew_energy'],
                             rew_foot_z=rew_components['rew_foot_z'],
                             penalty_torque_lim=rew_components['penalty_torque_lim'])

        # compute dones for resets
        done = self.is_done(new_norm_obs)

        # compute cmd for info
        _, cmd = self.torque_pd_control(scaled_action, prev_norm_obs)
        info = state.info
        info['cmd'] = cmd

        return state.replace(
            pipeline_state=new_pipeline_state, obs=new_norm_obs,
            reward=reward, done=done, u=scaled_action, info=info
        )

    def is_done(self, next_norm_obs: jp.ndarray) -> jp.ndarray:
        """Returns the done signal."""
        done = 1.0 - self._is_healthy(next_norm_obs)
        return done

    def _is_healthy(self, next_norm_obs: jp.ndarray) -> jp.ndarray:
        """Returns the healthy signal."""
        next_obs = self._denormalize_obs(next_norm_obs)
        roll, pitch, yaw = quat_to_eulerzyx(next_obs[self._quat_idxs])
        is_healthy = jp.where(
            jp.logical_and(jp.abs(roll) < self._fallen_roll,
                           jp.abs(pitch) < self._fallen_pitch),
            1.0, 0.0)
        return is_healthy

    def _update_system_properties(self, state: State):
        """Updates the system properties used for physics simulation with
        values that were set by the domain randomization"""
        sys = self.sys

        contact_time_const = state.info['contact_time_const']
        contact_damping_ratio = state.info['contact_damping_ratio']
        friction = state.info['friction']
        ground_roll = state.info['ground_roll']
        ground_pitch = state.info['ground_pitch']
        ground_quat = eulerzyx_to_quat(jp.array([ground_roll, ground_pitch, 0.0]))
        new_geoms = [
            g.replace(
                solver_params=g.solver_params.at[0, 0].set(contact_time_const)
            ) for g in sys.geoms
        ]
        new_geoms = [
            g.replace(
                solver_params=g.solver_params.at[0, 1].set(contact_damping_ratio)
            ) for g in new_geoms
        ]
        new_geoms = [
            g.replace(
                friction=g.friction.at[:].set(friction)
            ) for g in new_geoms
        ]
        new_geoms[0] = new_geoms[0].replace(
            transform=new_geoms[0].transform.replace(
                rot=new_geoms[0].transform.rot.at[0, :].set(ground_quat)
            )
        )

        joint_damping = state.info['joint_damping']
        new_dof = sys.dof.replace(
            damping=sys.dof.damping.at[:].set(joint_damping)
        )

        joint_gain = state.info['joint_gain']
        new_actuator = sys.actuator.replace(
            gain=sys.actuator.gain.at[:].set(joint_gain)
        )

        link_mass_perc = state.info['link_mass_perc']
        new_link = sys.link.replace(
            inertia=sys.link.inertia.replace(
                mass=sys.link.inertia.mass.at[:].set(
                    link_mass_perc * sys.link.inertia.mass
                ),
                i=sys.link.inertia.i.at[:, :, :].set(
                    jp.expand_dims(link_mass_perc, axis=(1, 2)) * sys.link.inertia.i
                )
            )
        )

        sys = sys.replace(geoms=new_geoms, dof=new_dof, actuator=new_actuator,
                          link=new_link)
        return sys

    def approx_dynamics(self, norm_obs: jp.ndarray, u: jp.ndarray,
                        ext_forces: Optional[jp.ndarray] = None,
                        norm_obs_next: Optional[jp.ndarray] = None) -> jp.ndarray:

        # u coming in is actually the scaled action (see output of self.step
        # funcion)
        scaled_action = u

        obs = self._denormalize_obs(norm_obs)
        obs_next = self._denormalize_obs(norm_obs_next)

        # get q and qd from obs
        q, qd = self.q_and_qd_from_obs(obs)

        pipeline_state_start = self._pipeline_init_approx(q, qd)

        def f(pipeline_state, _):
            # update obs with new q and qd from pipeline_state, but use the
            # same phase from obs
            norm_obs = self._get_obs_approx(pipeline_state, obs)
            torques, _ = self.torque_pd_control(scaled_action, norm_obs)
            pipeline_state = self._pipeline_step_approx(pipeline_state,
                                                        torques,
                                                        ext_forces)
            return pipeline_state, _

        pipeline_state, _ = jax.lax.scan(f, pipeline_state_start, (),
                                         self.policy_repeat)

        norm_obs_new = self._get_obs_approx(pipeline_state, obs_next)

        return norm_obs_new

    def dynamics_contact_integrate_only(
        self, norm_obs: jp.ndarray, u: jp.ndarray,
        ext_forces: Optional[jp.ndarray] = None,
        norm_obs_next: Optional[jp.ndarray] = None
    ) -> jp.ndarray:

        # u coming in is actually the scaled action (see output of self.step
        # funcion)
        scaled_action = u

        obs = self._denormalize_obs(norm_obs)
        obs_next = self._denormalize_obs(norm_obs_next)

        # get q and qd from obs
        q, qd = self.q_and_qd_from_obs(obs)

        # compute mass matrix and bias + passive forces
        sys = self.sys
        x, xd = kinematics.forward(sys, q, qd)
        state = GeneralizedState.init(q, qd, x, xd)
        state = dynamics.transform_com(sys, state)
        state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)
        state = state.replace(constraint_forces=jp.zeros_like(state.con_diag))
        qf_smooth = dynamics.forward(sys, state, jp.zeros(sys.qd_size()))
        qf_constraint = self._qfc_fn(state, ext_forces)
        state = state.replace(qf_constraint=qf_constraint)

        def f(state, _):
            # integrate q and qd, but with feedback from the torque pd control
            norm_obs = self._get_obs_approx(state, obs)
            torques, _ = self.torque_pd_control(scaled_action, norm_obs)
            tau = actuator.to_tau(sys, torques, state.q, state.qd)
            state = state.replace(qf_smooth=(qf_smooth + tau))
            state = integrator.integrate(sys, state)
            return state, None

        state, _ = jax.lax.scan(f, state, (), self.policy_repeat)

        norm_obs_new = self._get_obs_approx(state, obs_next)

        return norm_obs_new

    def exact_dynamics(self, norm_obs: jp.ndarray, u: jp.ndarray,
                       ext_forces: Optional[jp.ndarray] = None,
                       norm_obs_next: Optional[jp.ndarray] = None):
        # u coming in is actually the scaled action (see output of self.step
        # funcion)
        scaled_action = u

        obs = self._denormalize_obs(norm_obs)
        obs_next = self._denormalize_obs(norm_obs_next)

        # get q and qd from obs
        q, qd = self.q_and_qd_from_obs(obs)

        pipeline_state_start = self.pipeline_init(q, qd)

        def f(pipeline_state, _):
            norm_obs = self._get_obs_approx(pipeline_state, obs)
            u, _ = self.torque_pd_control(scaled_action, norm_obs)
            pipeline_state = pipeline_step(self.sys, pipeline_state, u)
            return pipeline_state, _

        pipeline_state, _ = jax.lax.scan(f, pipeline_state_start,
                                         (), self.policy_repeat)

        norm_obs_new = self._get_obs_approx(pipeline_state, obs_next)

        return norm_obs_new

    def mbpo_dynamics(self, norm_obs: jp.ndarray, u: jp.ndarray,
                      ext_forces: Optional[jp.ndarray] = None,
                      norm_obs_next: Optional[jp.ndarray] = None):
        norm_obs_new = norm_obs + ext_forces
        return norm_obs_new

    def q_and_qd_from_obs(self, obs: jp.ndarray):
        q = self.sys.init_q
        q = q.at[self._xml_quat_idxs].set(obs[self._quat_idxs])
        q = q.at[self._xml_q_idxs].set(obs[self._q_idxs])
        if self._include_height_in_obs:
            q = q.at[self._xml_h_idxs].set(obs[self._h_idx])

        quat = obs[self._quat_idxs]
        base_vel_body = obs[self._base_vel_idxs]
        base_vel_global = rotate(base_vel_body, quat)
        ang_vel_body = obs[self._rpy_rate_idxs]
        ang_vel_global = inv_rotate(ang_vel_body, quat)
        qd = jp.zeros(self.sys.qd_size())
        qd = qd.at[self._xml_base_vel_idxs].set(base_vel_global)
        qd = qd.at[self._xml_rpy_rate_idxs].set(ang_vel_global)
        qd = qd.at[self._xml_qd_idxs].set(obs[self._qd_idxs])

        return q, qd

    def make_ssrl_dynamics_fn(self, fn_type) -> Callable:

        fn = {
            'approx': self.approx_dynamics,
            'contact_integrate_only': self.dynamics_contact_integrate_only,
            'exact': self.exact_dynamics,
            'mbpo': self.mbpo_dynamics
        }[fn_type]

        def dynamics_fn(norm_obs: jp.ndarray, u: jp.ndarray, pred: jp.ndarray):

            obs_next = self._denormalize_obs(norm_obs)
            sin_phase = obs_next[self._sin_phase_idx]
            cos_phase = obs_next[self._cos_phase_idx]

            new_phase = (jp.arctan2(sin_phase, cos_phase)
                         + 2*jp.pi*self.sim_dt*self.policy_repeat/self._period)
            obs_next = obs_next.at[self._cos_phase_idx].set(jp.cos(new_phase))
            obs_next = obs_next.at[self._sin_phase_idx].set(jp.sin(new_phase))
            norm_obs_next = self._normalize_obs(obs_next)

            u = self.scale_action(u)

            norm_obs_new = fn(norm_obs, u, pred, norm_obs_next)

            return norm_obs_new

        return dynamics_fn

    def _get_obs(self, pipeline_state: base.State,
                 metrics: dict) -> jp.ndarray:
        """uses metrics to compute phase and desired velocity"""

        basic_obs = self._get_basic_obs(pipeline_state)

        # compute phase
        t = metrics['step_count'] * self.dt
        phase = 2*jp.pi*t/self._period
        cos_sin_phase = jp.array([jp.cos(phase), jp.sin(phase)])

        if self._include_height_in_obs:
            h = pipeline_state.q[self._xml_h_idxs]
            obs = jp.concatenate([basic_obs, cos_sin_phase, h])
        else:
            obs = jp.concatenate([basic_obs, cos_sin_phase])

        return self._normalize_obs(obs)

    def _get_obs_approx(self, pipeline_state: base.State,
                        obs_next: jp.ndarray) -> jp.ndarray:
        """uses the next observation to compute phase and desired velocity
        (this is OK since these obervations do not depend on the dynamics of
        the system)"""

        basic_obs = self._get_basic_obs(pipeline_state)

        cos_phase = obs_next[self._cos_phase_idx]
        sin_phase = obs_next[self._sin_phase_idx]
        cos_sin_phase = jp.array([cos_phase, sin_phase])

        if self._include_height_in_obs:
            h = pipeline_state.q[self._xml_h_idxs]
            obs = jp.concatenate([basic_obs, cos_sin_phase, h])
        else:
            obs = jp.concatenate([basic_obs, cos_sin_phase])

        return self._normalize_obs(obs)

    def _get_basic_obs(self, pipeline_state: base.State) -> jp.ndarray:
        "Returns basic observations without phase and desired velocity"

        positions = pipeline_state.q
        velocities = pipeline_state.qd

        # quat orientation of the base
        quat = positions[self._xml_quat_idxs]

        # joint angles
        q = positions[self._xml_q_idxs]

        # linear velocity of the base in the body frame
        base_vel_global = velocities[self._xml_base_vel_idxs]
        base_vel_body = inv_rotate(base_vel_global, quat)

        # angular velocity of the base in the body frame
        ang_vel_global = velocities[self._xml_rpy_rate_idxs]
        ang_vel_body = rotate(ang_vel_global, quat)

        # joint speeds
        qd = velocities[self._xml_qd_idxs]

        return jp.concatenate([quat, q, base_vel_body, ang_vel_body, qd])

    def _normalize_obs(self, obs: jp.ndarray) -> jp.ndarray:
        return (2*(obs - self.obs_limits[:, 0])
                / (self.obs_limits[:, 1] - self.obs_limits[:, 0])
                - 1)

    def _denormalize_obs(self, obs: jp.ndarray) -> jp.ndarray:
        return ((obs + 1)*(self.obs_limits[:, 1] - self.obs_limits[:, 0])/2
                + self.obs_limits[:, 0])

    def compute_reward(self, norm_obs: jp.ndarray, prev_norm_obs: jp.ndarray,
                       unused_u: jp.ndarray,
                       unscaled_action: jp.ndarray) -> jp.ndarray:
        obs = self._denormalize_obs(norm_obs)
        prev_obs = self._denormalize_obs(prev_norm_obs)

        scaled_action = self.scale_action(unscaled_action)
        u, cmd = self.torque_pd_control(scaled_action, prev_norm_obs,
                                        limit_Kp=False)

        reward, reward_components = self._reward_fn(obs, prev_obs, u,
                                                    unscaled_action, cmd)
        is_healthy = self._is_healthy(norm_obs)
        reward = reward + is_healthy - 1.0

        return reward, reward_components

    def ssrl_reward_fn(self, obs: jp.ndarray, obs_next: jp.ndarray,
                         unused_u: jp.ndarray,
                         unscaled_action: jp.ndarray) -> jp.ndarray:
        return self.compute_reward(obs_next, obs, unused_u,
                                   unscaled_action)[0]

    def _reward_normalized(self, obs: jp.ndarray,
                           prev_obs: jp.ndarray,
                           u: jp.ndarray,
                           unscaled_action: jp.ndarray,
                           cmd: ControlCommand) -> jp.ndarray:

        rew_forward_vel = obs[self._forward_vel_idx]
        # rew_forward_vel = -(obs[self._forward_vel_idx] - 2)**2 + 1
        rew_turn = jp.exp(
            -(obs[self._turn_rate_idx])**2/.2
        )
        roll, pitch, yaw = quat_to_eulerzyx(obs[self._quat_idxs])
        roll_prev, pitch_prev, yaw_prev = quat_to_eulerzyx(prev_obs[self._quat_idxs])
        rew_pitch = jp.exp(-pitch**2/.25)
        rew_roll = jp.exp(-roll**2/.25)
        rew_yaw = jp.exp(-yaw**2/0.07)
        rew_side_motion = jp.exp(
            -(obs[self._y_vel_idx])**2/.01
        )
        rew_z_vel_change = jp.exp(-(obs[self._z_vel_idx] - prev_obs[self._z_vel_idx])**2/0.02)
        rew_ang_vel = jp.exp(
            - (obs[self._roll_rate_idx])**2/.2
            - (obs[self._pitch_rate_idx])**2/.2
        )
        rew_ang_change = (
            jp.exp(-(roll - roll_prev)**2/0.001)
            + jp.exp(-(pitch - pitch_prev)**2/0.005)
        ) / 2
        rew_joint_limits = self._barrier_sigmoid(
            obs[self._q_idxs],
            jp.tile(Go1Utils.LOWER_JOINT_LIMITS, 4),
            jp.tile(Go1Utils.UPPER_JOINT_LIMITS, 4),
            w=10.0
        )
        rew_torque_limits = self._barrier_sigmoid(
            u,
            -.9*Go1Utils.MOTOR_TORQUE_LIMIT,
            .9*Go1Utils.MOTOR_TORQUE_LIMIT,
            w=0.2
        )
        rew_joint_acc = jp.exp(
            -jp.sum((obs[self._qd_idxs] - prev_obs[self._qd_idxs])**2)/4
        )
        reward_action = jp.mean(jp.exp(-unscaled_action**2/0.25))
        qstand = Go1Utils.ALL_STANDING_JOINT_ANGLES
        reward_cosmetic = jp.mean(jp.exp(-(obs[self._q_idxs] - qstand)**2/.05))
        reward_energy = jp.exp(-jp.sum(jp.abs(u*obs[self._qd_idxs]))**2/450)

        pdes = cmd.pdes
        p = Go1Utils.forward_kinematics_all_legs(obs[self._q_idxs])
        z_idxs = jp.array([2, 5, 8, 11])
        delta_zs = p[z_idxs] - pdes[z_idxs]
        reward_foot_z = jp.mean(jp.exp(-delta_zs**2/0.002))

        # penalty for exceeding torque limits, not normalized; exponential
        # until limit, and then linear above limit
        penalty_max = jax.lax.select(u > Go1Utils.MOTOR_TORQUE_LIMIT,
                                     -(u - Go1Utils.MOTOR_TORQUE_LIMIT + 1),
                                     -jp.exp(u-Go1Utils.MOTOR_TORQUE_LIMIT))
        penalty_min = jax.lax.select(u < -Go1Utils.MOTOR_TORQUE_LIMIT,
                                     (u + Go1Utils.MOTOR_TORQUE_LIMIT - 1),
                                     -jp.exp(-u-Go1Utils.MOTOR_TORQUE_LIMIT))
        penalty_torque_lim = jp.mean(penalty_max + penalty_min)

        rewards = jp.array([rew_forward_vel, rew_turn, rew_pitch, rew_roll,
                            rew_yaw, rew_side_motion, rew_z_vel_change,
                            rew_ang_vel, rew_ang_change,
                            rew_joint_limits, rew_torque_limits, rew_joint_acc,
                            reward_action, reward_cosmetic, reward_energy,
                            reward_foot_z, penalty_torque_lim])

        weighted_rewards = self._reward_weights * rewards

        reward_components = {'rew_forward_vel': weighted_rewards[0],
                             'rew_turn': weighted_rewards[1],
                             'rew_pitch': weighted_rewards[2],
                             'rew_roll': weighted_rewards[3],
                             'rew_yaw': weighted_rewards[4],
                             'rew_side_motion': weighted_rewards[5],
                             'rew_z_vel_change': weighted_rewards[6],
                             'rew_ang_vel': weighted_rewards[7],
                             'rew_ang_change': weighted_rewards[8],
                             'rew_joint_limits': weighted_rewards[9],
                             'rew_torque_limits': weighted_rewards[10],
                             'rew_joint_acc': weighted_rewards[11],
                             'rew_action': weighted_rewards[12],
                             'rew_cosmetic': weighted_rewards[13],
                             'rew_energy': weighted_rewards[14],
                             'rew_foot_z': weighted_rewards[15],
                             'penalty_torque_lim': weighted_rewards[16]}

        return jp.sum(weighted_rewards), reward_components

    def raw_reward_components(self, norm_obs: jp.ndarray,
                              prev_norm_obs: jp.ndarray,
                              unused_u: jp.ndarray,
                              unscaled_action: jp.ndarray) -> dict:
        """Exports raw reward components in order to calculate means and stds
        for normalization"""

        obs = self._denormalize_obs(norm_obs)
        prev_obs = self._denormalize_obs(prev_norm_obs)
        scaled_action = self.scale_action(unscaled_action)
        u, cmd = self.torque_pd_control(scaled_action, prev_norm_obs, limit_Kp=False)

        forward_vel_err = obs[self._forward_vel_idx]
        turn_rate_err = obs[self._turn_rate_idx]
        quat = obs[self._quat_idxs]
        roll, pitch, yaw = quat_to_eulerzyx(quat)
        roll_prev, pitch_prev, yaw_prev = quat_to_eulerzyx(prev_obs[self._quat_idxs])
        y_vel = obs[self._y_vel_idx]
        z_vel_change = obs[self._z_vel_idx] - prev_obs[self._z_vel_idx]
        roll_change = roll - roll_prev
        pitch_change = pitch - pitch_prev

        energy = jp.sum(jp.abs(u*obs[self._qd_idxs]))

        # penalty for exceeding torque limits, not normalized
        buf = 2.0
        torque_limit_over = jp.where(u > Go1Utils.MOTOR_TORQUE_LIMIT - buf,
                                     u - (Go1Utils.MOTOR_TORQUE_LIMIT - buf),
                                     0.0)
        torque_limit_under = jp.where(u < -Go1Utils.MOTOR_TORQUE_LIMIT + buf,
                                      -u - (Go1Utils.MOTOR_TORQUE_LIMIT - buf),
                                      0.0)
        exceeded_torques = jp.maximum(torque_limit_over, torque_limit_under)

        return {
            'forward_vel_err': forward_vel_err,
            'turn_rate_err': turn_rate_err,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'y_vel': y_vel,
            'z_vel_change': z_vel_change,
            'roll_change': roll_change,
            'pitch_change': pitch_change,
            'energy': energy,
            'exceeded_torques': exceeded_torques
        }

    def _barrier_sigmoid(self, x: jp.ndarray, x_min: jp.ndarray,
                         x_max: jp.ndarray, w: float = 10.) -> jp.ndarray:
        """A smooth function made up of two sigmoids which returns 1 if x is
        between x_min and x_max, returns values closer to zero as x approaches
        x_min or x_max. Use w to adjust the width of the sigmoid."""
        sig = (1/(1 + jp.exp(-w*(x - x_min)))
               + 1/(1 + jp.exp(w*(x - x_max))) - 1)
        return jp.mean(sig)

    def _quadratic_limit(self, x: jp.ndarray, x_min: jp.ndarray,
                         x_max: jp.ndarray) -> jp.ndarray:
        rew = jp.zeros_like(x)
        rew = jp.where(x < x_min, -(x - x_min)**2, rew)
        rew = jp.where(x > x_max, -(x - x_max)**2, rew)
        return jp.mean(rew)

    def _quadratic_limit_with_buffer(self, x: jp.ndarray, x_min: jp.ndarray,
                                     x_max: jp.ndarray,
                                     buf: jp.ndarray) -> jp.ndarray:
        rew = jp.zeros_like(x)
        rew = jp.where(x < x_min + buf, -(x/buf - x_min/buf - 1)**2, rew)
        rew = jp.where(x > x_max - buf, -(x/buf - x_max/buf + 1)**2, rew)
        return jp.mean(rew)

    def _z_score(self, x: jp.ndarray, x_mean: jp.ndarray,
                 x_std: jp.ndarray) -> jp.ndarray:
        return (x - x_mean)/x_std

    def _linear_limit(self, x: jp.ndarray, x_min: jp.ndarray,
                      x_max: jp.ndarray) -> jp.ndarray:
        rew = jp.zeros_like(x)
        rew = jp.where(x < x_min, -jp.abs(x - x_min), rew)
        rew = jp.where(x > x_max, -jp.abs(x - x_max), rew)
        return jp.mean(rew)

    def _huber(self, x: jp.ndarray, w: jp.ndarray) -> jp.ndarray:
        """Huber reward function for x with width w"""
        return jp.where(jp.abs(x) < w, -0.5*x**2, -jp.abs(x) + 0.5*w)

    def _weighed_huber(self, x: jp.ndarray, w: jp.ndarray) -> jp.ndarray:
        """Weighed Huber reward function for x with width w"""
        return jp.where(jp.abs(x) < jp.sqrt(2*w),
                        -x**2/(2*w) + 1.0,
                        -1/w*jp.sqrt(2*w)*jp.abs(x) + 2)

    def _weighed_quadratic(self, x: jp.ndarray, w: jp.ndarray) -> jp.ndarray:
        """Weighed quadratic reward function for x with width w"""
        return -x**2/(2*w) + 1.0

    def _smooth_abs(self, x: jp.ndarray) -> jp.ndarray:
        """Smooth absolute value function for x with width w"""
        return -jp.linalg.norm(x)**2 / (jp.linalg.norm(x) + 1)

    def _constant_fwd_vel(self, metrics: dict) -> jp.ndarray:
        return metrics['forward_vel_des']

    def _sine_fwd_vel(self, metrics: dict) -> jp.ndarray:
        a = self._forward_cmd_vel_range[0]
        b = self._forward_cmd_vel_range[1]
        T = metrics['forward_vel_period']
        phase = metrics['forward_vel_phase']
        t = metrics['step_count'] * self.dt
        return (b - a)/2 * jp.sin(2*jp.pi*t/T + phase) + (b + a)/2

    def torque_pd_control(self, action: jp.ndarray,
                          norm_obs: jp.ndarray,
                          limit_Kp: bool = True) -> Tuple[jp.ndarray, ControlCommand]:

        obs = self._denormalize_obs(norm_obs)

        cmd = self.low_level_control_hardware(action, obs, limit_Kp=limit_Kp)

        # torque control
        q = obs[self._q_idxs]
        qd = obs[self._qd_idxs]
        u = cmd.Kp*(cmd.q_des - q) + cmd.Kd*(cmd.qd_des - qd)

        return u, cmd

    def low_level_control(self, scaled_action: jp.ndarray,
                          unused_norm_obs: jp.ndarray) -> jp.ndarray:
        # Here we simply return the action as we are treating as "control
        # inputs" to the env the actions. Low level PD torque control is
        # instead absorbed into the appoximate dynamics.
        return scaled_action

    def low_level_control_hardware(self, action: jp.ndarray,
                                   obs: jp.ndarray,
                                   limit_Kp: bool = True) -> ControlCommand:
        # gait control
        dbody_h = -0.05
        if self._body_height_in_action_space:
            dbody_h = action[self._ac_dbody_h_idx]
        gait_params = Go1GaitParams(
            period=self._period,
            r=0.5,
            swing_h=0.09,
            dbody_h=dbody_h,
            bias=jp.array([0.0, 0.5, 0.5, 0.0])
        )
        pdes, contact, leg_phases = Go1Gait.control(
            gait_params=gait_params,
            forward_vel_des=self._forward_cmd_vel * jp.ones(()),
            turn_rate_des=jp.zeros(()),
            cos_phase=obs[self._cos_phase_idx],
            sin_phase=obs[self._sin_phase_idx]
        )

        delta_xy = action[self._ac_delta_pdes_idxs]
        deltas = jp.array([
            delta_xy[0], delta_xy[4], 0.0,
            delta_xy[1], delta_xy[5], 0.0,
            delta_xy[2], delta_xy[6], 0.0,
            delta_xy[3], delta_xy[7], 0.0,
        ])
        pdes = pdes + deltas

        # inverse kinematics
        q_des = Go1Utils.inverse_kinematics_all_legs(pdes)
        q_des = jp.clip(q_des,
                        jp.tile(Go1Utils.LOWER_JOINT_LIMITS, 4),
                        jp.tile(Go1Utils.UPPER_JOINT_LIMITS, 4))
        qd_des = jp.zeros((12,))
        mult = 1.4
        Kp = jp.tile(jp.array([80, 80, 80]), 4) * mult
        Kd = jp.tile(jp.array([2.5, 2.5, 2.5]), 4)

        if self._gains_in_action_space:
            Kp += action[self._ac_Kp_idxs]
            Kd += action[self._ac_Kd_idxs]

        Kp = jax.lax.cond(limit_Kp,
                          self._limit_Kp,
                          lambda *args: Kp,
                          obs, q_des, qd_des, Kp, Kd)

        return ControlCommand(q_des, qd_des, Kp, Kd, contact, leg_phases, pdes)

    def _limit_Kp(self, obs, q_des, qd_des, Kp, Kd):
        # limits kp if torque is too high
        q = obs[self._q_idxs]
        qd = obs[self._qd_idxs]
        torque_theo = Kp*(q_des - q) + Kd*(qd_des - qd)
        q_err = jp.where(q_des - q != 0, q_des - q, 1)  # avoid div by zero
        Kp = jp.where(
            torque_theo > Go1Utils.MOTOR_TORQUE_LIMIT,
            (Go1Utils.MOTOR_TORQUE_LIMIT - Kd*(qd_des - qd))/q_err, Kp)
        Kp = jp.where(
            torque_theo < -Go1Utils.MOTOR_TORQUE_LIMIT,
            (-Go1Utils.MOTOR_TORQUE_LIMIT - Kd*(qd_des - qd))/q_err, Kp)
        return Kp

    def _pipeline_init_approx(self, q: jp.ndarray,
                              qd: jp.ndarray) -> base.State:
        """Initializes the pipeline state for the approximate system."""
        sys = self._sys_approx
        x, xd = kinematics.forward(sys, q, qd)
        state = GeneralizedState.init(q, qd, x, xd)
        state = dynamics.transform_com(sys, state)
        state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)
        state = state.replace(constraint_forces=jp.zeros_like(state.con_diag))
        return state

    def _pipeline_step_approx(
        self, pipeline_state: Any, act: jp.ndarray, ext_forces: jp.ndarray
    ) -> base.State:
        """Takes a physics step using the physics pipeline on the approximate
        system, but with custom contact forces."""

        sys = self._sys_approx

        def f(state, _):
            # calculate acceleration terms
            tau = actuator.to_tau(sys, act, state.q, state.qd)
            state = state.replace(qf_smooth=dynamics.forward(sys, state, tau))
            qf_constraint = self._qfc_fn(state, ext_forces)
            state = state.replace(qf_constraint=qf_constraint)

            # update position/velocity level terms
            state = integrator.integrate(sys, state)
            # x, xd = kinematics.forward(sys, state.q, state.qd)
            # state = state.replace(x=x, xd=xd)
            # state = dynamics.transform_com(sys, state)
            # state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)

            return state, None

        return jax.lax.scan(f, pipeline_state, (), self._n_frames)[0]

    def scale_action(self, unscaled_action: jp.ndarray) -> jp.ndarray:
        return ((self._ac_space[:, 1] - self._ac_space[:, 0])*unscaled_action/2
                + (self._ac_space[:, 1] + self._ac_space[:, 0])/2)

    def unscale_action(self, scaled_action: jp.ndarray) -> jp.ndarray:
        return (2*(scaled_action - (self._ac_space[:, 1] + self._ac_space[:, 0])/2)
                / (self._ac_space[:, 1] - self._ac_space[:, 0]))

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def controls_size(self) -> int:
        return self._action_size

    @property
    def observation_size(self) -> int:
        return self._observation_size

    @property
    def dt(self) -> jp.ndarray:
        """The timestep used for each env step."""
        return self.sim_dt * self.policy_repeat

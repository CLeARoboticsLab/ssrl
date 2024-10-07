from brax.robots.go1.utils import Go1Utils
from brax.math import quat_to_3x3
from unitree_legged_msgs.msg import LowState
from ssrl_ros_go1_msgs.msg import Observation
from ssrl_ros_go1_msgs.msg import Gait
from ssrl_ros_go1_msgs.msg import Estimation
from ssrl_ros_go1_msgs.msg import QuadrupedState
from brax.envs.go1_deltaxy_pd_slow import Go1DeltaXyPdSlow as Go1

from flax import struct
from jax import numpy as jp
from jax.scipy import linalg as scla
import rospy
import jax
import numpy as np


@struct.dataclass
class EstimatorState:
    xhat: jp.ndarray            # state estimate, velocity(3) (global frame)
    vel_body: jp.ndarray        # velocity (3) (body frame)
    P: jp.ndarray               # prediction covariance


@struct.dataclass
class EstimatorObservation:
    quat: jp.ndarray            # quaternion of body in global frame (4)
    q: jp.ndarray               # joint angles (12)
    ang_vel_body: jp.ndarray    # angular velocity of body in body frame (3)
    qd: jp.ndarray              # joint velocities (12)
    contact: jp.ndarray         # nominal contact state, bool (4)
    leg_phases: jp.ndarray           # phase of each foot, float (4)
    accel: jp.ndarray           # acceleration of body in body frame (3)


@struct.dataclass
class EstimatorParameters:
    A: jp.ndarray               # state transition matrix
    B: jp.ndarray               # control input matrix
    C: jp.ndarray               # output matrix
    QInit: jp.ndarray           # process noise covariance
    RInit: jp.ndarray           # initial measurement noise covariance
    g: jp.ndarray               # gravity vector
    largeVariance: jp.ndarray   # large variance for when leg is swinging


class Estimator:
    def __init__(self):
        rospy.init_node('estimator')

        self.rate = 25
        Qdig = 0.0003 # adjustable prcoess noise covariance

        rospy.loginfo("Building environment...")
        self.env = Go1(used_cached_systems=True)
        self._quat_idxs = self.env._quat_idxs
        self._q_idxs = self.env._q_idxs
        self._rpy_rate_idxs = self.env._rpy_rate_idxs
        self._qd_idxs = self.env._qd_idxs

        self.dt = 1/self.rate
        self.obs = jp.zeros(38)
        self.obs = self.obs.at[self._quat_idxs].set(jp.array([1., 0., 0., 0.]))
        self.contact = jp.array([True, True, True, True])
        self.leg_phases = jp.zeros(4)
        self.accel = jp.zeros(3)

        A = jp.eye(3)              
        B = self.dt * jp.eye(3)    
        C = jp.tile(-jp.eye(3), (4, 1))    
        RInit = jp.array([                     
            [1.708,  0.048,  0.784,  0.062,  0.042,  0.053,  0.077,  0.001, -0.061,  0.046, -0.019, -0.029],
            [0.048 , 5.001 ,-1.631 ,-0.036 , 0.144 , 0.040 , 0.036 , 0.016 ,-0.051 ,-0.067 ,-0.024 ,-0.005],
            [0.784, -1.631,  1.242,  0.057, -0.037,  0.018,  0.034, -0.017, -0.015,  0.058, -0.021, -0.029],
            [0.062 ,-0.036 , 0.057 , 6.228 ,-0.014 , 0.932 , 0.059 , 0.053 ,-0.069 , 0.148 , 0.015 ,-0.031],
            [0.042,  0.144, -0.037, -0.014,  3.011,  0.986,  0.076,  0.030, -0.052, -0.027,  0.057,  0.051],
            [0.053,  0.040,  0.018,  0.932,  0.986,  0.885,  0.090,  0.044, -0.055,  0.057,  0.051, -0.003],
            [0.077,  0.036,  0.034,  0.059,  0.076,  0.090,  6.230,  0.139,  0.763,  0.013, -0.019, -0.024],
            [0.001 , 0.016 ,-0.017 , 0.053 , 0.030 , 0.044 , 0.139 , 3.130 ,-1.128 ,-0.010 , 0.131 , 0.018],
            [-0.061, -0.051, -0.015, -0.069, -0.052, -0.055,  0.763, -1.128,  0.866, -0.022, -0.053,  0.007], 
            [ 0.046, -0.067,  0.058,  0.148, -0.027,  0.057,  0.013, -0.010, -0.022,  2.437, -0.102,  0.938], 
            [-0.019, -0.024, -0.021,  0.015,  0.057,  0.051, -0.019,  0.131, -0.053, -0.102,  4.944,  1.724], 
            [-0.029, -0.005, -0.029, -0.031,  0.051, -0.003, -0.024,  0.018,  0.007,  0.938,  1.724,  1.569]
        ])
        Cu = jp.array([     # control input noise covariance       
            [268.573,   -43.819, -147.211],
            [-43.819 ,   92.949 ,  58.082],
            [-147.211,   58.082,  302.120]
        ])
        QInit = Qdig * jp.eye(3) + B @ Cu @ B.T
        g = jp.array([0., 0., -9.81])
        self.largeVariance = 100.

        self.params = EstimatorParameters(A=A, B=B, C=C, QInit=QInit,
                                          RInit=RInit, g=g,
                                          largeVariance=self.largeVariance)
        
        # run estimation once to jit compile
        self.qped_state = 'off'
        self.reset_state()
        est_obs = self.get_estimator_observation()
        estimate(self.state, est_obs, self.params)
        
        self.obs_sub = rospy.Subscriber('observation', Observation,
                                        self.obs_callback)
        self.gait_sub = rospy.Subscriber('gait', Gait,
                                         self.gait_callback)
        self.low_sub = rospy.Subscriber('low_state', LowState,
                                        self.low_callback)
        self.qped_state_sub = rospy.Subscriber('quadruped_state', QuadrupedState,
                                               self.qped_state_callback)
        self.est_pub = rospy.Publisher('estimation', Estimation, queue_size=100)

    def reset_state(self):
        self.state = EstimatorState(xhat=jp.zeros(3),
                                    vel_body=jp.zeros(3),
                                    P=self.largeVariance*jp.eye(3))
    
    def obs_callback(self, observation: Observation):
        self.obs = jp.array(observation.observation)

    def gait_callback(self, gait: Gait):
        self.contact = jp.array(gait.contact)
        self.leg_phases = jp.array(gait.leg_phases)
        
    def low_callback(self, low_state: LowState):
        self.accel = jp.array(low_state.imu.accelerometer)

    def qped_state_callback(self, qped_state: QuadrupedState):
        self.qped_state = qped_state.state
        if self.qped_state == 'off' or self.qped_state == 'stand' or self.qped_state == 'standing_up':
            self.reset_state()
            self.publish_estimation()

    def publish_estimation(self):
        msg = Estimation()
        msg.linear_vel = np.array(self.state.vel_body)
        msg.linear_vel_std_dev = np.sqrt(np.array(self.state.P).diagonal())
        self.est_pub.publish(msg)

    def get_estimator_observation(self):
        return EstimatorObservation(
                    quat=self.obs[self._quat_idxs],
                    q=self.obs[self._q_idxs],
                    ang_vel_body=self.obs[self._rpy_rate_idxs],
                    qd=self.obs[self._qd_idxs],
                    contact=self.contact,
                    leg_phases=self.leg_phases,
                    accel=self.accel
                )

    def run(self):
        rospy.loginfo("Estimator started")
        rate = rospy.Rate(1/self.dt)
        
        while not rospy.is_shutdown():
            est_obs = self.get_estimator_observation()
            self.state = estimate(self.state, est_obs, self.params)
            self.publish_estimation()
            
            rate.sleep()


@jax.jit
def estimate(state: EstimatorState, obs: EstimatorObservation,
             p: EstimatorParameters):
    
    # compute foot velocities wrt to body in global frame
    feetVel = Go1Utils.foot_vel_all_legs(obs.q, obs.qd)
    feetPos = Go1Utils.forward_kinematics_all_legs(obs.q)
    w = skew(obs.ang_vel_body)
    feetVel += scla.block_diag(w,w,w,w) @ feetPos
    rot = quat_to_3x3(obs.quat)
    feetVel_wrt_body_in_global = scla.block_diag(rot, rot, rot, rot) @ feetVel

    # determine measurement noise covariance based on nominal contact state
    Q = p.QInit
    R = p.RInit
    contact_FR = obs.contact[0]
    contact_FL = obs.contact[1]
    contact_RR = obs.contact[2]
    contact_RL = obs.contact[3]
    lv = p.largeVariance * jp.eye(3)
    trust = window(obs.leg_phases, 0.2*jp.ones(4), jp.ones(4), jp.ones(4))
    added_cov = 1 + (1-trust)*p.largeVariance
    R_c_FR = added_cov[0] + p.RInit[0:3, 0:3]
    R_c_FL = added_cov[1] + p.RInit[3:6, 3:6]
    R_c_RR = added_cov[2] + p.RInit[6:9, 6:9]
    R_c_RL = added_cov[3] + p.RInit[9:12, 9:12]
    R = R.at[0:3, 0:3].set(jax.lax.select(contact_FR, R_c_FR, lv))
    R = R.at[3:6, 3:6].set(jax.lax.select(contact_FL, R_c_FL, lv))
    R = R.at[6:9, 6:9].set(jax.lax.select(contact_RR, R_c_RR, lv))
    R = R.at[9:12, 9:12].set(jax.lax.select(contact_RL, R_c_RL, lv))

    # Kalman filter
    u = rot @ obs.accel + p.g
    xbar = p.A @ state.xhat + p.B @ u
    yhat = p.C @ xbar
    y = feetVel_wrt_body_in_global
    Pbar = p.A @ state.P @ p.A.T + Q
    S = p.C @ Pbar @ p.C.T + R
    K = Pbar @ p.C.T @ jp.linalg.solve(S, jp.eye(12))
    xhat = xbar + K @ (y - yhat)
    P = (jp.eye(3) - K @ p.C) @ Pbar

    vel_body = rot.T @ xhat

    return EstimatorState(xhat=xhat, vel_body=vel_body, P=P)


def skew(x: jp.ndarray):
    return jp.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])


def window(x, window_ratio, x_range, y_range):
    out = y_range
    out = jp.where(x/x_range < window_ratio,
                   x * y_range / (x_range * window_ratio),
                   out)
    out = jp.where(x/x_range > 1 - window_ratio,
                y_range * (x_range - x)/(x_range * window_ratio),
                out)
    return out

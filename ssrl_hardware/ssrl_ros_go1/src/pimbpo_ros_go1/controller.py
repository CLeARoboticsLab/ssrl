from ssrl_ros_go1_msgs.msg import Observation
from ssrl_ros_go1_msgs.msg import Action
from ssrl_ros_go1_msgs.msg import PdTarget
from ssrl_ros_go1_msgs.msg import Gait
from ssrl_ros_go1_msgs.msg import QuadrupedState

from ssrl_ros_go1 import env_dict
from brax.envs.go1_deltaxyk_pd_slow import ControlCommand as Cmd
from brax.robots.go1.utils import Go1Utils
from brax.robots.go1 import networks as go1_networks
from brax.math import quat_to_eulerzyx, eulerzyx_to_quat

from omegaconf import DictConfig
from pathlib import Path
import jax
from jax import config
config.update("jax_enable_x64", True)
from jax import numpy as jp
import numpy as np
import rospy
import rosbag
from pynput import keyboard
import threading
import math
import re
import os


class Controller:
    def __init__(self, cfg: DictConfig, data_path: Path):
        assert cfg.run_name is not None

        # parameters
        self.obs_history_length = cfg.common.obs_history_length
        self.action_repeat = 1
        self.use_policy = True
        self.save_data = True
        mult = 0.25
        self.Kp_stand_start = jp.tile(jp.array([70, 70, 80]), 4) * mult
        self.Kd_stand_start = jp.tile(jp.array([5, 5, 7]), 4) * mult
        self.Kp_stand_end = jp.tile(jp.array([70, 70, 80]), 4)
        self.Kd_stand_end = jp.tile(jp.array([5, 5, 7]), 4)
        self.p_stand_end = Go1Utils.standing_foot_positions()
        self.p_stand_end += jp.tile(jp.array([0, 0, 0.05]), 4)
        self.p_stand_start = self.p_stand_end + jp.tile(jp.array([0, 0, 0.22]), 4)
        self.standing_up_time = 5.0
        self.lp_alpha = 1.0

        # environment
        rospy.init_node('controller')
        rospy.loginfo(f"Building {cfg.env} environment...")
        self.env = env_dict[cfg.env](
            used_cached_systems=True,
            policy_repeat=cfg.env_common.policy_repeat)
        self.obs_size = self.env.observation_size
        self.obs = jp.zeros(self.obs_size)
        self.obs_copy = jp.copy(self.obs_size)
        self.obs_msg = Observation()
        self.action = jp.zeros(self.env.action_size)
        self.action_msg = Action()
        self.norm_obs_stack = jp.zeros(self.obs_size*self.obs_history_length)
        self.control = jax.jit(self.env.low_level_control_hardware)
        self.normalize_obs = jax.jit(self.env._normalize_obs)
        self.scale_action = jax.jit(self.env.scale_action)
        
        # Class vars
        self.data_path = data_path
        self.bag = None
        self.subrollout_count = 0
        self.step_count = 0
        self.rollout_step_count = 0
        self.standing_up_count = 0
        self.dt = self.env.dt
        self.start_yaw = 0.0
        self.is_straight_task = cfg.env == 'Go1GoFast'
        self.ik = jax.jit(Go1Utils.inverse_kinematics_all_legs)
        self.last_cmd = Cmd(jp.zeros(12,), jp.zeros(12,),
                            jp.zeros(12,), jp.zeros(12,),
                            jp.array([True]*4), jp.array([0.0]*4),
                            jp.zeros(12,))
        self.is_done = jax.jit(self.env.is_done)
        
        # precompile jitted functions
        quat = jp.array([1, 0, 0, 0])
        yaw_from_quat(quat)
        quat_from_start_yaw(quat, 0.0)
        self.scale_action(jp.zeros(self.env.action_size))
        self.ik(self.p_stand_end)
        self.is_done(self.obs)

        if os.path.exists(self.data_path / cfg.run_name):
            # continue a run
            self.rollout_count = max(
                [int(folder) for folder in os.listdir(
                    self.data_path / cfg.run_name)]) + 1
            sac_ts_path = (self.data_path / cfg.run_name
                           / f"{self.rollout_count-1:02d}" / "sac_ts.pkl")
            params, make_policy, _ = go1_networks.make_sac_networks(
                cfg, self.env, saved_policies_dir=None,
                sac_ts_path=sac_ts_path)
            self.max_rollout_steps = cfg.ssrl.env_steps_per_training_step
        else:
            # start a new run
            if self.save_data:
                os.makedirs(self.data_path / cfg.run_name)
            self.rollout_count = 0
            params, make_policy, _ = go1_networks.make_sac_networks(
                cfg, self.env, saved_policies_dir=None, sac_ts_path=None,
                seed=cfg.ssrl.seed)
            self.max_rollout_steps = cfg.ssrl.init_exploration_steps

        rospy.loginfo(f"Starting rollout {self.rollout_count:02d}")

        self.rollout_path = self.data_path / cfg.run_name / f"{self.rollout_count:02d}"
        if self.save_data and not os.path.exists(self.rollout_path):
            os.makedirs(self.rollout_path)

        if self.use_policy:
            policy = make_policy(params,
                                 deterministic=cfg.ssrl.deterministic_in_env)
        else:
            policy = lambda x, y: (jp.zeros(self.env.action_size), {})

        self.policy = jax.jit(policy)
        self.key = jax.random.PRNGKey(cfg.ssrl.seed)
        self.scaled_action = jp.zeros(self.env.action_size)

        # run control once to compile
        self.qped_state = 'walk'
        self.do_control(publish=False)
        self.qped_state = 'off'

        # pd_target messages
        self.off_pd_target = PdTarget()
        self.off_pd_target.mode = 0x00
        self.off_pd_target.q_des[:12] = [math.pow(10,9)] * 12
        self.off_pd_target.qd_des[:12] = [16000.0] * 12
        self.off_pd_target.Kp[:12] = [0] * 12
        self.off_pd_target.Kd[:12] = [0] * 12
        self.pd_target = PdTarget()
        self.pd_target.mode = 0x0A
        self.pd_target.q_des[:12] = [math.pow(10,9)] * 12
        self.pd_target.qd_des[:12] = [16000.0] * 12
        self.pd_target.Kp[:12] = [0] * 12
        self.pd_target.Kd[:12] = [0] * 12

        # subscribers and publishers
        self.obs_sub = rospy.Subscriber('observation', Observation,
                                        self.obs_callback,
                                        tcp_nodelay=True)
        self.pd_target_pub = rospy.Publisher("pd_target", PdTarget,
                                             queue_size=100)
        self.gait_pub = rospy.Publisher("gait", Gait, queue_size=100)
        self.qped_state_pub = rospy.Publisher("quadruped_state", QuadrupedState,
                                               queue_size=10)

        # keyboard listener
        self.listener_thread = threading.Thread(target=self.start_keyboard_listener)
        self.listener_thread.start()

    def obs_callback(self, observation: Observation):
        self.obs = jp.array(observation.observation)

    def do_control(self, publish=True):
        # create a copy to prevent mutation from the callback
        self.obs_copy = jp.copy(self.obs)

        # rotate the quaternion to be relative to the start yaw (straight task
        # only)
        if self.is_straight_task:
            new_quat = quat_from_start_yaw(self.obs_copy[self.env._quat_idxs],
                                        self.start_yaw)
            self.obs_copy = self.obs_copy.at[self.env._quat_idxs].set(new_quat)

        norm_obs = self.normalize_obs(self.obs_copy)
        self.norm_obs_stack = jp.concatenate(
            [norm_obs, self.norm_obs_stack[:self.obs_size*(self.obs_history_length-1)]],
            axis=-1
        )
        if self.qped_state != 'off':
            if self.qped_state == 'walk':
                if self.step_count % self.action_repeat == 0:
                    self.key, key_act = jax.random.split(self.key)
                    self.action = self.policy(self.norm_obs_stack, key_act)[0]
                    self.scaled_action = self.scale_action(self.action)
                cmd = self.control(self.scaled_action, self.obs_copy)
                q_des = cmd.q_des
                qd_des = cmd.qd_des
                Kp = cmd.Kp
                Kd = cmd.Kd
                contact = cmd.contact
                leg_phases = cmd.leg_phases
                self.step_count += 1
                self.rollout_step_count += 1
            elif self.qped_state == 'standing_up':
                x = self.standing_up_count/ (self.standing_up_time / self.dt)
                Kp = self.interpolate(self.Kp_stand_start, self.Kp_stand_end, x)
                Kd = self.interpolate(self.Kd_stand_start, self.Kd_stand_end, x)
                p_des = self.interpolate(self.p_stand_start, self.p_stand_end, x)
                q_des = self.ik(p_des)
                q_des = jp.clip(q_des,
                                jp.tile(Go1Utils.LOWER_JOINT_LIMITS, 4),
                                jp.tile(Go1Utils.UPPER_JOINT_LIMITS, 4))
                qd_des = jp.zeros((12,))
                contact = jp.array([True]*4)
                leg_phases = jp.array([0.0]*4)
                if self.standing_up_count >= self.standing_up_time / self.dt:
                    self.qped_state = 'stand'
                    self.publish_quadruped_state()
                    self.standing_up_count = 0
                    rospy.loginfo("Quadruped is standing. Press space to walk or any key to turn off.")
                else:
                    self.standing_up_count += 1
            elif self.qped_state == 'stand':
                p_des = self.p_stand_end
                q_des = self.ik(p_des)
                q_des = jp.clip(q_des,
                                jp.tile(Go1Utils.LOWER_JOINT_LIMITS, 4),
                                jp.tile(Go1Utils.UPPER_JOINT_LIMITS, 4))
                qd_des = jp.zeros((12,))
                Kp = self.Kp_stand_end
                Kd = self.Kd_stand_end
                contact = jp.array([True]*4)
                leg_phases = jp.array([0.0]*4)
                self.last_cmd = Cmd(q_des, qd_des, Kp, Kd, contact, leg_phases,
                                    jp.zeros(12,))
            if publish:
                self.publish_pd_target(q_des, qd_des, Kp, Kd)
                self.publish_gait(contact, leg_phases)
            if self.qped_state == 'walk':
                self.write_bag()
                self.check_termination(norm_obs)
        else:
            if publish:
                self.publish_offcmd()

    def interpolate(self, start: jp.ndarray, end: jp.ndarray, xs: jp.ndarray):
        """Interpolate between two values where xs are between 0
        and 1."""
        xs = jp.clip(xs, 0, 1)
        ys = start + (end - start) * xs
        return ys

    def publish_pd_target(self, q_des: jp.ndarray, qd_des: jp.ndarray,
                     Kp: jp.ndarray, Kd: jp.ndarray):
        
        q_des = self.lp_alpha*q_des + (1-self.lp_alpha)*self.last_cmd.q_des
        qd_des = self.lp_alpha*qd_des + (1-self.lp_alpha)*self.last_cmd.qd_des
        Kp = self.lp_alpha*Kp + (1-self.lp_alpha)*self.last_cmd.Kp
        Kd = self.lp_alpha*Kd + (1-self.lp_alpha)*self.last_cmd.Kd
        
        self.pd_target.q_des[:12] = np.array(q_des)
        self.pd_target.qd_des[:12] = np.array(qd_des)
        self.pd_target.Kp[:12] = np.array(Kp)
        self.pd_target.Kd[:12] = np.array(Kd)
        self.pd_target_pub.publish(self.pd_target)

        self.last_cmd = Cmd(q_des, qd_des, Kp, Kd,
                            jp.array([True]*4), jp.array([0.0]*4),
                            jp.zeros(12,))

    def publish_gait(self, contact: jp.ndarray, leg_phases: jp.ndarray):
        msg = Gait()
        msg.contact = np.array(contact)
        msg.leg_phases = np.array(leg_phases)
        self.gait_pub.publish(msg)

    def publish_offcmd(self):
        self.pd_target_pub.publish(self.off_pd_target)

    def publish_quadruped_state(self):
        msg = QuadrupedState()
        msg.state = self.qped_state
        self.qped_state_pub.publish(msg)

    def start_keyboard_listener(self):
        with keyboard.Listener(on_press=self.on_press) as self.listener:
            self.listener.join()

    def on_press(self, key):
        if key == keyboard.Key.space:
            if self.qped_state == "walk":
                self.qped_state = "stand"
                self.close_bag()
                rospy.loginfo("Quadruped is standing. Press space to walk or any key to turn off.")
            elif self.qped_state == "off":
                rospy.loginfo("Quadruped is standing up...")
                self.standing_up_count = 0
                self.qped_state = "standing_up"
            elif self.qped_state == "stand":
                rospy.loginfo("Quadruped is walking. Press space to stand or any key to turn off.")
                self.step_count = 0
                if self.is_straight_task:
                    self.start_yaw = yaw_from_quat(self.obs[self.env._quat_idxs])
                # Don't count steps when the obs hist is still filling up
                self.max_rollout_steps += self.obs_history_length + 1
                self.open_bag()
                self.qped_state = "walk"
        else:
            self.qped_state = "off"
            self.close_bag()
            rospy.loginfo("Quadruped is off. Press space to start standing up.")
        self.publish_quadruped_state()

    def check_termination(self, norm_obs):
        if (not self.is_done(norm_obs)
                and self.rollout_step_count < self.max_rollout_steps):
            return
        
        self.publish_offcmd()
        self.qped_state = "off"
        self.close_bag()
        self.publish_quadruped_state()
        if self.rollout_step_count >= self.max_rollout_steps:
            rospy.loginfo("Max rollout steps reached.")
            rospy.loginfo("Quadruped is off.")
        elif self.is_done(norm_obs):
            rospy.loginfo("Robot is not healthy. Terminated subrollout.")
            rospy.loginfo("Quadruped is off. Press space to start standing up.")

    def open_bag(self):
        if not self.save_data:
            return
        
        # delete old subrollouts if this is the first subrollout
        if self.subrollout_count == 0:
            pattern = re.compile(r'subrollout_\d+\.bag')
            for file in os.listdir(self.rollout_path):
                if pattern.match(file):
                    os.remove(os.path.join(self.rollout_path, file))

        # open new bag
        bag_name = 'subrollout_{:0>2}.bag'.format(self.subrollout_count)
        self.bag = rosbag.Bag(os.path.join(self.rollout_path, bag_name), 'w')

    def write_bag(self):
        if self.bag is not None and self.action is not None:
            self.obs_msg.observation = np.array(self.obs_copy)
            self.action_msg.action = np.array(self.action)
            self.bag.write('observation', self.obs_msg)
            self.bag.write('pd_target', self.pd_target)
            self.bag.write('action', self.action_msg)

    def close_bag(self):
        if self.bag is not None:
            self.bag.close()
            self.subrollout_count += 1
            rospy.loginfo(f"Rollout total steps: {self.rollout_step_count}")
        self.bag = None

    def run(self):
        rospy.loginfo("Starting controller")
        rate = rospy.Rate(1/self.env.dt)
        rospy.loginfo("Quadruped is off. Press space to start standing up.")
        self.publish_quadruped_state()

        while not rospy.is_shutdown():
            self.do_control()
            rate.sleep()

        self.qped_state = "off"
        self.do_control()
        self.close_bag()
        print("Shutdown detected, press any key to exit.")
        self.listener.stop()
        self.listener_thread.join()


@jax.jit
def yaw_from_quat(quat: jp.ndarray):
    r, p, y = quat_to_eulerzyx(quat)
    return y


@jax.jit
def quat_from_start_yaw(abs_quat: jp.ndarray, start_yaw: float):
    r, p, abs_yaw = quat_to_eulerzyx(abs_quat)
    rel_yaw = abs_yaw - start_yaw
    rel_yaw = jp.arctan2(jp.sin(rel_yaw), jp.cos(rel_yaw)) # wrap to [-pi, pi]
    new_rpy_deg = jp.array([r, p, rel_yaw]) * 180 / jp.pi
    return eulerzyx_to_quat(new_rpy_deg)
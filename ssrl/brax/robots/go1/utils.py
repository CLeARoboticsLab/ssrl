from brax.base import System
from etils import epath
from brax.io import mjcf
from jax import numpy as jp
import jax
import dill
from pathlib import Path


class Go1Utils:
    """Utility functions for the Go1."""

    """
    Properties
    """
    THIGH_OFFSET = 0.08
    """constant: the length of the thigh motor"""

    LEG_OFFSET_X = 0.1881
    """constant: x distance from the robot COM to the leg base."""

    LEG_OFFSET_Y = 0.04675
    """constant: y distance from the robot COM to the leg base."""

    THIGH_LENGTH = 0.213
    """constant: length of the thigh and also the length of the calf"""

    CALF_LENGTH = 0.213
    """constant: length of the calf"""

    STANDING_FOOT_POSITIONS = jp.array([
        0.1881, -0.1300, -0.3200,
        0.1881, 0.1300, -0.3200,
        -0.1881, -0.1300, -0.3200,
        -0.1881, 0.1300, -0.3200
    ])

    STANDING_JOINT_ANGLES_FR = jp.array([-0.01014303, 0.7180088, -1.4360176])
    STANDING_JOINT_ANGLES_FL = jp.array([0.01014303, 0.7180088, -1.4360176])
    STANDING_JOINT_ANGLES_RR = jp.array([-0.01014303, 0.7180088, -1.4360176])
    STANDING_JOINT_ANGLES_RL = jp.array([0.01014303, 0.7180088, -1.4360176])

    ALL_STANDING_JOINT_ANGLES = jp.concatenate([
        STANDING_JOINT_ANGLES_FR,
        STANDING_JOINT_ANGLES_FL,
        STANDING_JOINT_ANGLES_RR,
        STANDING_JOINT_ANGLES_RL
    ])

    JOINT_LIMIT_PAD = 0.1
    """constant: the amount to pad the joint limits"""

    LOWER_JOINT_LIMITS = jp.array([-1.047, -0.663, -2.721]) + JOINT_LIMIT_PAD
    """constant: the lower joint angle limits for a leg, obtained from
    unitree_legged_sdk/include/go1_const.h, and offset by JOINT_LIMIT_PAD"""

    UPPER_JOINT_LIMITS = jp.array([1.047, 2.966, -0.837]) - JOINT_LIMIT_PAD
    """constant: the upper joint angle limits for a leg, obtained from
    unitree_legged_sdk/include/go1_const.h, and offset by JOINT_LIMIT_PAD"""

    MOTOR_TORQUE_LIMIT = jp.tile(jp.array([23.7, 23.7, 35.55]), 4)
    """constant: the torque limit for the motors"""

    CACHE_PATH = epath.resource_path('brax') / 'robots/go1/.cache'

    @staticmethod
    def get_system(used_cached: bool = False) -> System:
        """Returns the system for the Go1."""

        if used_cached:
            sys = Go1Utils._load_cached_system(approx_system=False)
        else:
            # load in urdf file
            path = epath.resource_path('brax')
            path /= 'robots/go1/xml/go1.xml'
            sys = mjcf.load(path)

        return sys

    @staticmethod
    def get_approx_system(used_cached: bool = False) -> System:
        """Returns the approximate system for the Go1."""

        if used_cached:
            sys = Go1Utils._load_cached_system(approx_system=True)
        else:
            # load in urdf file
            path = epath.resource_path('brax')
            path /= 'robots/go1/xml/go1_approx.xml'
            sys = mjcf.load(path)

        return sys

    @staticmethod
    def _cache_system(approx_system: bool) -> System:
        """Cache the system for the Go1 to avoid reloading the xml file."""
        sys = Go1Utils.get_system()
        Path(Go1Utils.CACHE_PATH).mkdir(parents=True, exist_ok=True)
        with open(Go1Utils._cache_path(approx_system), 'wb') as f:
            dill.dump(sys, f)
        return sys

    @staticmethod
    def _load_cached_system(approx_system: bool) -> System:
        """Load the cached system for the Go1."""
        try:
            with open(Go1Utils._cache_path(approx_system), 'rb') as f:
                sys = dill.load(f)
        except FileNotFoundError:
            sys = Go1Utils._cache_system(approx_system)
        return sys

    @staticmethod
    def _cache_path(approx_system: bool) -> epath.Path:
        """Get the path to the cached system for the Go1."""
        if approx_system:
            path = Go1Utils.CACHE_PATH / 'go1_approx_system.pkl'
        else:
            path = Go1Utils.CACHE_PATH / 'go1_system.pkl'
        return path

    @staticmethod
    def forward_kinematics(leg: str, q: jp.ndarray) -> jp.ndarray:
        """Returns the position of the foot in the body frame centered on the
           trunk, given the joint angles; (3,)

        Arguments:
            leg (str): the name of the leg - 'FR', 'FL', 'RR', 'RL'
            q (jp.ndarray): the joint angles of a leg; (3,)
        """
        if leg not in ['FR', 'FL', 'RR', 'RL']:
            raise ValueError('leg must be one of FR, FL, RR, RL')

        side_sign = jax.lax.select(leg in ['FR', 'RR'], -1, 1)

        l1 = side_sign * Go1Utils.THIGH_OFFSET
        l2 = -Go1Utils.THIGH_LENGTH
        l3 = -Go1Utils.CALF_LENGTH

        s1 = jp.sin(q[0])
        s2 = jp.sin(q[1])
        s3 = jp.sin(q[2])

        c1 = jp.cos(q[0])
        c2 = jp.cos(q[1])
        c3 = jp.cos(q[2])

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        p0_hip = l3 * s23 + l2 * s2
        p1_hip = -l3 * s1 * c23 + l1 * c1 - l2 * c2 * s1
        p2_hip = l3 * c1 * c23 + l1 * s1 + l2 * c1 * c2

        p0 = p0_hip + jax.lax.select(leg in ['FR', 'FL'],
                                     Go1Utils.LEG_OFFSET_X,
                                     -Go1Utils.LEG_OFFSET_X)
        p1 = p1_hip + jax.lax.select(leg in ['FR', 'RR'],
                                     -Go1Utils.LEG_OFFSET_Y,
                                     Go1Utils.LEG_OFFSET_Y)
        p2 = p2_hip

        p = jp.stack([p0, p1, p2], axis=0)
        return p

    @staticmethod
    def forward_kinematics_all_legs(q: jp.ndarray) -> jp.ndarray:
        """Returns the positions of the feet in the body frame centered on the
           trunk, given the joint angles; (12,)

        Arguments:
            q (jp.ndarray): the joint angles of all legs; (12,)
        """
        p = jp.concatenate([
            Go1Utils.forward_kinematics('FR', q[0:3]),
            Go1Utils.forward_kinematics('FL', q[3:6]),
            Go1Utils.forward_kinematics('RR', q[6:9]),
            Go1Utils.forward_kinematics('RL', q[9:12]),
        ])
        return p

    @staticmethod
    def inverse_kinematics(leg: str, p: jp.ndarray) -> jp.ndarray:
        """Returns the joint angles of a leg given the position of the foot in
           the body frame centered on the trunk; (3,)

        Arguments:
            leg (str): the name of the leg - 'FR', 'FL', 'RR', 'RL'
            p (jp.ndarray): the position of the foot in the body frame; (3,)
        """
        if leg not in ['FR', 'FL', 'RR', 'RL']:
            raise ValueError('leg must be one of FR, FL, RR, RL')

        fx = jax.lax.select(leg in ['RR', 'RL'],
                            -Go1Utils.LEG_OFFSET_X,
                            Go1Utils.LEG_OFFSET_X)
        fy = jax.lax.select(leg in ['FR', 'RR'],
                            -Go1Utils.LEG_OFFSET_Y,
                            Go1Utils.LEG_OFFSET_Y)

        px = p[0] - fx  # TODO: double check
        py = p[1] - fy
        pz = p[2]

        b2y = jax.lax.select(leg in ['FR', 'RR'],
                             -Go1Utils.THIGH_OFFSET,
                             Go1Utils.THIGH_OFFSET)
        b3z = -Go1Utils.THIGH_LENGTH
        b4z = -Go1Utils.THIGH_LENGTH
        a = Go1Utils.THIGH_OFFSET
        c = jp.sqrt(px**2 + py**2 + pz**2)
        b = jp.sqrt(c**2 - a**2)

        L = jp.sqrt(py**2 + pz**2 - b2y**2)
        q1 = jp.arctan2(pz*b2y+py*L, py*b2y-pz*L)

        temp = (b3z**2 + b4z**2 - b**2)/(2*jp.abs(b3z*b4z))
        q3max = Go1Utils.UPPER_JOINT_LIMITS[2]
        q3min = Go1Utils.LOWER_JOINT_LIMITS[2]
        # instead of clipping withing -1 and 1, clip per the below to ensure q3
        # stays within joint limits (and also prevent nan gradients)
        temp = jp.clip(temp, jp.cos(jp.pi+q3max), jp.cos(jp.pi+q3min))
        q3 = jp.arccos(temp)
        q3 = -(jp.pi - q3)

        a1 = py*jp.sin(q1) - pz*jp.cos(q1)
        a2 = px
        m1 = b4z*jp.sin(q3)
        m2 = b3z + b4z*jp.cos(q3)
        q2 = jp.arctan2(m1*a1+m2*a2, m1*a2-m2*a1)

        q = jp.stack([q1, q2, q3], axis=0)
        return q

    @staticmethod
    def inverse_kinematics_all_legs(p: jp.ndarray) -> jp.ndarray:
        """Returns the joint angles of all legs given the positions of the feet
           in the body frame centered on the trunk; (12,)

        Arguments:
            p (jp.ndarray): the positions of the feet in the body frame; (12,)
        """
        q = jp.concatenate([
            Go1Utils.inverse_kinematics('FR', p[0:3]),
            Go1Utils.inverse_kinematics('FL', p[3:6]),
            Go1Utils.inverse_kinematics('RR', p[6:9]),
            Go1Utils.inverse_kinematics('RL', p[9:12]),
        ])
        return q

    @staticmethod
    def jacobian(leg: str, q: jp.ndarray) -> jp.ndarray:
        """get the jacobian of the leg

        Arguments:
            leg (str): the name of the leg - 'FR', 'FL', 'RR', 'RL'
            q (jp.ndarray): the joint angles of a leg; (3,)

        Returns:
            jp.ndarray: the jacobian of the leg, (3, 3)
        """

        if leg not in ['FR', 'FL', 'RR', 'RL']:
            raise ValueError('leg must be one of FR, FL, RR, RL')

        d = jax.lax.select(leg in ['FR', 'RR'],
                           -Go1Utils.THIGH_OFFSET,
                           Go1Utils.THIGH_OFFSET)
        length = Go1Utils.THIGH_LENGTH

        q1 = q[0]
        q2 = q[1]
        q3 = q[2]

        J00 = 0.
        J01 = -length*(jp.cos(q2 + q3) + jp.cos(q2))
        J02 = -length*jp.cos(q2 + q3)
        J10 = (
            length*jp.cos(q1)*jp.cos(q2)
            - d*jp.sin(q1)
            + length*jp.cos(q1)*jp.cos(q2)*jp.cos(q3)
            - length*jp.cos(q1)*jp.sin(q2)*jp.sin(q3)
        )
        J11 = -length*jp.sin(q1)*(jp.sin(q2 + q3) + jp.sin(q2))
        J12 = -length*jp.sin(q2 + q3)*jp.sin(q1)
        J20 = (
            d*jp.cos(q1)
            + length*jp.cos(q2)*jp.sin(q1)
            + length*jp.cos(q2)*jp.cos(q3)*jp.sin(q1)
            - length*jp.sin(q1)*jp.sin(q2)*jp.sin(q3)
        )
        J21 = length*jp.cos(q1)*(jp.sin(q2 + q3) + jp.sin(q2))
        J22 = length*jp.sin(q2 + q3)*jp.cos(q1)

        J = jp.stack([
            jp.stack([J00, J01, J02], axis=0),
            jp.stack([J10, J11, J12], axis=0),
            jp.stack([J20, J21, J22], axis=0)
        ], axis=0)

        return J

    @staticmethod
    def foot_vel(leg: str, q: jp.ndarray, qd: jp.ndarray) -> jp.ndarray:
        """Returns the linear velocity of the foot in the body frame; (3,)

        Arguments:
            leg (str): the name of the leg - 'FR', 'FL', 'RR', 'RL'
            q (jp.jp.ndarray): the joint angles of a leg; (3,)
            qd (jp.jp.ndarray): the joint speeds of a leg; (3,)
        """
        J = Go1Utils.jacobian(leg, q)
        vel = jp.matmul(J, qd)
        return vel

    def foot_vel_all_legs(q: jp.ndarray, qd: jp.ndarray) -> jp.ndarray:
        """Returns the linear velocities of all feet in the body frame; (12,)

        Arguments:
            q (jp.ndarray): the joint angles of all legs; (12,)
            qd (jp.ndarray): the joint speeds of all legs; (12,)
        """
        vel = jp.concatenate([
            Go1Utils.foot_vel('FR', q[0:3], qd[0:3]),
            Go1Utils.foot_vel('FL', q[3:6], qd[3:6]),
            Go1Utils.foot_vel('RR', q[6:9], qd[6:9]),
            Go1Utils.foot_vel('RL', q[9:12], qd[9:12]),
        ])
        return vel

    @staticmethod
    def standing_foot_positions() -> jp.ndarray:
        """Returns the positions of the feet in the body frame when the robot
        is standing; (12,)"""
        return Go1Utils.STANDING_FOOT_POSITIONS


if __name__ == '__main__':
    q = jp.array([0.1, 0.2, 0.3])
    qd = jp.array([0.1, 0.2, 0.3])
    p = Go1Utils.forward_kinematics('FR', q)
    pd = Go1Utils.foot_vel('FR', q, qd)
    J = Go1Utils.jacobian('FR', q)
    print(p)
    print(pd)
    print(J)
    print(Go1Utils.standing_foot_positions())
    print(Go1Utils.inverse_kinematics_all_legs(
        Go1Utils.standing_foot_positions()))

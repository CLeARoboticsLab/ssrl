from brax.robots.go1.utils import Go1Utils
from flax import struct
from jax import numpy as jp
import jax


STANDING_FEET_POS = Go1Utils.standing_foot_positions()
FR_STAND = STANDING_FEET_POS[0:3]
FL_STAND = STANDING_FEET_POS[3:6]
RR_STAND = STANDING_FEET_POS[6:9]
RL_STAND = STANDING_FEET_POS[9:12]
FOOT_RAD = jp.sqrt(Go1Utils.LEG_OFFSET_X**2
                   + (Go1Utils.LEG_OFFSET_Y + Go1Utils.THIGH_OFFSET)**2)
foot_delta_idxs = {
    'FR': jp.s_[0:2],
    'FL': jp.s_[2:4],
    'RR': jp.s_[4:6],
    'RL': jp.s_[6:8]
}


@struct.dataclass
class Go1GaitParams:
    period: jp.ndarray   # period of gait (sec)
    r: jp.ndarray        # fraction of gait spent in contact with ground
    swing_h: jp.ndarray  # height of foot above ground during swing (m)
    dbody_h: jp.ndarray  # height of body above ground, delta from standing (m)
    bias: jp.ndarray     # bias for each leg; shape (4,)


class Go1Gait:

    @staticmethod
    def control(gait_params: Go1GaitParams,
                forward_vel_des: jp.ndarray,
                turn_rate_des: jp.ndarray,
                cos_phase: jp.ndarray,
                sin_phase: jp.ndarray,
                foot_fall_deltas: jp.ndarray = jp.zeros((8,))) -> jp.ndarray:
        """Compute desired foot positions in the body frame. All arguments will
        need to be formatted in the order FR, FL, RR, RL.

        Arguments:
            gait_params: gait parameters
            forward_vel_des: desired forward velocity (m/s), shape ()
            turn_rate_des: desired turn rate (rad/s), shape ()
            cos_phase: cosine of the phase variable, shape ()
            sin_phase: sine of the phase variable, shape ()
        """

        swing_h = gait_params.swing_h

        # compute phase variable and desired contact state for each leg
        phase = jp.arctan2(sin_phase, cos_phase)
        phi_FR, contact_FR = _phi(phase, gait_params.bias[0], gait_params.r)
        phi_FL, contact_FL = _phi(phase, gait_params.bias[1], gait_params.r)
        phi_RR, contact_RR = _phi(phase, gait_params.bias[2], gait_params.r)
        phi_RL, contact_RL = _phi(phase, gait_params.bias[3], gait_params.r)

        # compute the ideal foot position at the start of the swing phase
        t_swing = (1 - gait_params.r)*gait_params.period
        start_swing_pos_FR = _start_swing_pos('FR', forward_vel_des,
                                              turn_rate_des, t_swing)
        start_swing_pos_FL = _start_swing_pos('FL', forward_vel_des,
                                              turn_rate_des, t_swing)
        start_swing_pos_RR = _start_swing_pos('RR', forward_vel_des,
                                              turn_rate_des, t_swing)
        start_swing_pos_RL = _start_swing_pos('RL', forward_vel_des,
                                              turn_rate_des, t_swing)

        # this is also the ideal position at the end of the contact phase
        end_contact_pos_FR = start_swing_pos_FR
        end_contact_pos_FL = start_swing_pos_FL
        end_contact_pos_RR = start_swing_pos_RR
        end_contact_pos_RL = start_swing_pos_RL

        # compute the desired foot posistion at the end of the swing phase
        end_swing_pos_FR = _end_swing_pos('FR', forward_vel_des,
                                          turn_rate_des, t_swing,
                                          foot_fall_deltas)
        end_swing_pos_FL = _end_swing_pos('FL', forward_vel_des,
                                          turn_rate_des, t_swing,
                                          foot_fall_deltas)
        end_swing_pos_RR = _end_swing_pos('RR', forward_vel_des,
                                          turn_rate_des, t_swing,
                                          foot_fall_deltas)
        end_swing_pos_RL = _end_swing_pos('RL', forward_vel_des,
                                          turn_rate_des, t_swing,
                                          foot_fall_deltas)

        # this is also the ideal position at the start of the contact phase
        start_contact_pos_FR = end_swing_pos_FR
        start_contact_pos_FL = end_swing_pos_FL
        start_contact_pos_RR = end_swing_pos_RR
        start_contact_pos_RL = end_swing_pos_RL

        # compute the change in foot position during the swing phase
        foot_pos_change_swing_FR = _foot_pos_change_swing(
            swing_h, phi_FR, start_swing_pos_FR, end_swing_pos_FR)
        foot_pos_change_swing_FL = _foot_pos_change_swing(
            swing_h, phi_FL, start_swing_pos_FL, end_swing_pos_FL)
        foot_pos_change_swing_RR = _foot_pos_change_swing(
            swing_h, phi_RR, start_swing_pos_RR, end_swing_pos_RR)
        foot_pos_change_swing_RL = _foot_pos_change_swing(
            swing_h, phi_RL, start_swing_pos_RL, end_swing_pos_RL)

        # compute the change in foot position during the contact phase
        foot_pos_change_contact_FR = _foot_pos_change_contact(
            phi_FR, start_contact_pos_FR, end_contact_pos_FR)
        foot_pos_change_contact_FL = _foot_pos_change_contact(
            phi_FL, start_contact_pos_FL, end_contact_pos_FL)
        foot_pos_change_contact_RR = _foot_pos_change_contact(
            phi_RR, start_contact_pos_RR, end_contact_pos_RR)
        foot_pos_change_contact_RL = _foot_pos_change_contact(
            phi_RL, start_contact_pos_RL, end_contact_pos_RL)

        # select the desired foot position based on the contact state
        foot_pos_FR = FR_STAND + jax.lax.select(contact_FR,
                                                foot_pos_change_contact_FR,
                                                foot_pos_change_swing_FR)
        foot_pos_FL = FL_STAND + jax.lax.select(contact_FL,
                                                foot_pos_change_contact_FL,
                                                foot_pos_change_swing_FL)
        foot_pos_RR = RR_STAND + jax.lax.select(contact_RR,
                                                foot_pos_change_contact_RR,
                                                foot_pos_change_swing_RR)
        foot_pos_RL = RL_STAND + jax.lax.select(contact_RL,
                                                foot_pos_change_contact_RL,
                                                foot_pos_change_swing_RL)

        dh = jp.tile(jp.array([0.0, 0.0, gait_params.dbody_h]), 4)
        pdes = jp.concatenate([foot_pos_FR, foot_pos_FL,
                               foot_pos_RR, foot_pos_RL]) - dh
        contact = jp.array([contact_FR, contact_FL,
                            contact_RR, contact_RL])
        leg_phases = jp.array([phi_FR, phi_FL, phi_RR, phi_RL])

        return pdes, contact, leg_phases


def _time_fraction(phase, bias):
    return jp.mod(phase/(2*jp.pi) + 0.5 + bias, 1)


def _phi(phase, bias, r):
    time_fraction = _time_fraction(phase, bias)
    contact = jax.lax.select(time_fraction < r, True, False)
    phase = jax.lax.select(contact,
                           time_fraction / r,
                           (time_fraction - r) / (1 - r))
    return phase, contact


def _start_swing_pos(leg, forward_vel_des, turn_rate_des, t_swing):
    return -_end_swing_pos(leg, forward_vel_des, turn_rate_des, t_swing,
                           jp.zeros((8,)))


def _end_swing_pos(leg, forward_vel_des, turn_rate_des, t_swing,
                   foot_fall_deltas):
    if leg not in ['FR', 'FL', 'RR', 'RL']:
        raise ValueError('leg must be one of FR, FL, RR, RL')
    x_stand = jax.lax.select(leg in ['FR', 'FL'], FR_STAND[0], RR_STAND[0])
    y_stand = jax.lax.select(leg in ['FR', 'RR'], FR_STAND[1], FL_STAND[1])
    dtheta = turn_rate_des*t_swing
    dx_turn = FOOT_RAD*jp.cos(jp.arctan2(y_stand, x_stand) + dtheta) - x_stand
    dy_turn = FOOT_RAD*jp.sin(jp.arctan2(y_stand, x_stand) + dtheta) - y_stand
    dx_foot_fall = foot_fall_deltas[foot_delta_idxs[leg]][0]
    dy_foot_fall = foot_fall_deltas[foot_delta_idxs[leg]][1]
    return jp.array([forward_vel_des*t_swing + dx_turn + dx_foot_fall,
                     dy_turn + dy_foot_fall,
                     0.0])


def _foot_pos_change_swing(h, phi, start_swing_pos, end_swing_pos):
    return jp.array([
        _cycloid_xy(phi, start_swing_pos[0], end_swing_pos[0]),
        _cycloid_xy(phi, start_swing_pos[1], end_swing_pos[1]),
        h/2 * (1 - jp.cos(2*jp.pi*phi))
    ])


def _foot_pos_change_contact(phi, start_contact_pos, end_contact_pos):
    return jp.array([
        _cycloid_xy(phi, start_contact_pos[0], end_contact_pos[0]),
        _cycloid_xy(phi, start_contact_pos[1], end_contact_pos[1]),
        0.0
    ])


def _cycloid_xy(phi, start, end):
    return (end - start)/(2*jp.pi)*(2*jp.pi*phi - jp.sin(2*jp.pi*phi)) + start


if __name__ == "__main__":
    gait_params = Go1GaitParams(
        period=0.5,
        r=0.5,
        swing_h=0.08,
        dbody_h=-0.05,
        bias=jp.array([0.0, 0.5, 0.5, 0.0])
    )
    forward_vel_des = jp.array(0.0)
    turn_rate_des = jp.array(0.0)
    phase = jp.pi/6
    cos_phase = jp.array(jp.cos(phase))
    sin_phase = jp.array(jp.sin(phase))
    pdes = Go1Gait.control(gait_params, forward_vel_des, turn_rate_des,
                           cos_phase, sin_phase)[0]
    print(pdes)

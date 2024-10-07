from omegaconf import OmegaConf
from datetime import datetime
import wandb
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jp
from brax.robots.go1.utils import Go1Utils
import jax


def plot_rollout(cfg, env, ts, 
                 obses=None, qs=None, qds=None, q_deses=None, qd_deses=None,
                 Kps=None, Kds=None, actions=None, theo_torques=None,
                 theo_energy=None, rollout_num=None, init_wandb=True):
    if init_wandb:
        run_name = cfg.algo + '_' + datetime.now().strftime("%Y-%m-%d_%H%M_%S")
        if cfg.run_name is not None:
            run_name = cfg.run_name

        config_dict = OmegaConf.to_container(cfg, resolve=True,
                                                throw_on_missing=True)
        wandb.init(project='go1_hardware',
                    entity=cfg.wandb.entity,
                    name=run_name,
                    config=config_dict)
        print(OmegaConf.to_yaml(cfg))

    actions = jp.array(actions)
    actions = np.array(env.scale_action(actions))
    obses = np.array(obses)
    q_deses = np.array(q_deses)
    qd_deses = np.array(qd_deses)
    Kps = np.array(Kps)
    Kds = np.array(Kds)
    theo_torques = np.array(theo_torques)
    theo_energy = np.array(theo_energy)

    if rollout_num is not None:
        prefix = 'rollout_{:0>2}_'.format(rollout_num)
    else:
        prefix = ''

    _generate_wandb_plot(ts, actions, prefix + 'action')
    _generate_wandb_plot(ts, obses, prefix + 'observation')
    _generate_wandb_plot(ts, q_deses, prefix + 'q_des')
    _generate_wandb_plot(ts, qd_deses, prefix + 'qd_des')
    _generate_wandb_plot(ts, Kps, prefix + 'Kp')
    _generate_wandb_plot(ts, Kds, prefix + 'Kd')
    _generate_wandb_plot(ts, theo_torques, prefix + 'theo_torque')
    _generate_wandb_plot(ts, theo_energy, prefix + 'theo_energy')

    _plot_foot_positions(env, ts, obses, q_deses)


def _generate_wandb_plot(ts, data, name):
    if (data == None).any():
        return
    for i in range(data.shape[1]):
        fig, ax = plt.subplots()
        ax.plot(ts, data[:, i])
        ax.set(xlabel='time (s)', ylabel=name, title=name+'_{:0>2}'.format(i))
        wandb.log({name+'_{:0>2}'.format(i): plt})
        plt.clf()

def _plot_foot_positions(env, ts, obses, q_deses):
    fk = jax.vmap(Go1Utils.forward_kinematics_all_legs)
    
    qs = jp.array(obses)[:, env._q_idxs]
    ps = np.array(fk(qs))
    p_deses = np.array(fk(jp.array(q_deses)))

    for i in range(ps.shape[1]):
        fig, ax = plt.subplots()
        ax.plot(ts, ps[:, i], label='p')
        ax.plot(ts, p_deses[:, i], label='pdes')
        name="foot_pos"
        ax.set(xlabel='time (s)', ylabel=name, title=name+'_{:0>2}'.format(i))
        wandb.log({name+'_{:0>2}'.format(i): plt})
        plt.clf()    

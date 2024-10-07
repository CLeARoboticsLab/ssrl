from omegaconf import DictConfig
import hydra
import os
import functools as ft
import dill

from brax import actuator
from brax import kinematics
from brax.generalized.base import State as GeneralizedState
from brax.generalized import dynamics
from brax.generalized import mass


@hydra.main(config_path="configs", config_name="go1")
def validate_model(cfg: DictConfig):
    hardware_data = cfg.torque_validate.hardware_data

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if hardware_data:
        data_path = os.path.join(base_path, 'saved_data/model_validation/hardware_data/ssrl_data.pkl')
        sac_ts_path = os.path.join(base_path, 'saved_data/model_validation/hardware_data/sac_ts.pkl')
        save_path = os.path.join(base_path, 'saved_data/model_validation/hardware_data/pred_and_real_forces.pkl')
        model_path = None
    else:
        data_path = os.path.join(base_path, 'saved_data/model_validation/sim_data/ssrl_state.pkl')
        model_path = os.path.join(base_path, 'saved_data/model_validation/sim_data/training_state.pkl')
        save_path = os.path.join(base_path, 'saved_data/model_validation/sim_data/pred_and_real_forces.pkl')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)

    # use 64-bit precision to limit rollout dynamics mismatch
    from jax import config
    config.update("jax_enable_x64", True)
    # config.update("jax_debug_nans", True)  # REMOVE

    from brax.training.agents.ssrl import train as ssrl
    from brax.envs.go1_go_fast import Go1GoFast
    from brax.robots.go1 import networks as go1_networks
    from brax.training.agents.ssrl import base
    from jax import numpy as jp
    import jax


    # create env fn
    env_dict = {'Go1GoFast': Go1GoFast}
    env_fn = ft.partial(env_dict[cfg.env], backend='generalized')
    env_kwargs = cfg.env_ssrl
    env_fn = add_kwargs_to_fn(env_fn, **env_kwargs)
    env = env_fn(used_cached_systems=True)

    for k, v in cfg.ssrl.items():
        if v == 'None' or v == 'none':
            cfg.ssrl[k] = None
    for k, v in cfg.actor_network.items():
        if v == 'None' or v == 'none':
            cfg.actor_network[k] = None

    dynamics_fn = env.make_ssrl_dynamics_fn(cfg.ssrl_dynamics_fn)

    # make networks
    (sac_network_factory,
        model_network_factory) = go1_networks.ssrl_network_factories(cfg)

    init_train_fn = ft.partial(ssrl.initizalize_training)
    init_train_fn = add_kwargs_to_fn(init_train_fn, **cfg.ssrl)
    model_horizon_fn = ssrl.make_linear_threshold_fn(
        cfg.ssrl_linear_threshold_fn.start_epoch,
        cfg.ssrl_linear_threshold_fn.end_epoch,
        cfg.ssrl_linear_threshold_fn.start_model_horizon,
        cfg.ssrl_linear_threshold_fn.end_model_horizon)
    hupts_fn = ssrl.make_linear_threshold_fn(
        cfg.ssrl_hupts_fn.start_epoch,
        cfg.ssrl_hupts_fn.end_epoch,
        cfg.ssrl_hupts_fn.start_hupts,
        cfg.ssrl_hupts_fn.end_hupts)
    output_dim = (env.observation_size if cfg.ssrl_dynamics_fn == 'mbpo'
                  else env.sys.qd_size())
    ms = init_train_fn(
        environment=env,
        low_level_control_fn=env.low_level_control,
        dynamics_fn=dynamics_fn,
        reward_fn=env.ssrl_reward_fn,
        model_output_dim=output_dim,
        model_horizon_fn=model_horizon_fn,
        hallucination_updates_per_training_step=hupts_fn,
        sac_training_state=None,
        model_network_factory=model_network_factory,
        sac_network_factory=sac_network_factory,
        num_timesteps=0,
        progress_fn=lambda *args: None,
        policy_params_fn=lambda *args: None,
        eval_env=env
    )

    if hardware_data:
        with open(data_path, 'rb') as f:
            ssrl_data = dill.load(f)
        ts = ssrl_data.ssrl_state
        env_buffer_state = ssrl_data.env_buffer_state
        with open(sac_ts_path, 'rb') as f:
            sac_ts = dill.load(f)
    else:
        with open(data_path, 'rb') as f:
            (ts, sac_ts, env_buffer_state) = dill.load(f)

    ms = ms.replace(
        sac_state=ms.sac_state.replace(training_state=sac_ts),
        env_buffer_state=env_buffer_state
    )

    buffer_size = ms.env_buffer.size(env_buffer_state)
    data = env_buffer_state.data[:buffer_size]
    all_transitions = ms.env_buffer._unflatten_fn(data)

    start = 17000
    stop = 18000
    obs_stack = all_transitions.observation[start:stop]
    next_obs_stack = all_transitions.next_observation[start:stop]
    actions = all_transitions.action[start:stop]
    time = jp.arange(0, (stop-start)/100, 0.01)

    # generate force predictions
    model_network = model_network_factory(ms.constants.obs_size, output_dim)
    preprocess_fn = base.Scaler.transform
    proc_obs_stack, proc_act_r = preprocess_fn(obs_stack, actions, 
                                               ts.scaler_params)
    x = jp.concatenate([proc_obs_stack, proc_act_r], axis=-1)
    x = jp.expand_dims(x, axis=1)
    x = jp.tile(x, reps=(1, 7, 1))

    def apply_fn(x):
        return model_network.apply(ts.model_params, x)
    preds, logvars = jax.vmap(apply_fn)(x)
    preds = jp.mean(preds, axis=1)

    # compute real forces
    norm_obs = obs_stack[:, :ms.constants.obs_size]
    norm_obs_next = next_obs_stack[:, :ms.constants.obs_size]
    scaled_action = env.scale_action(actions)

    def est_constraint_forces(norm_obs, norm_obs_next, scaled_action):
        obs = env._denormalize_obs(norm_obs)
        obs_next = env._denormalize_obs(norm_obs_next)

        q, qd = env.q_and_qd_from_obs(obs)
        q_next, qd_next = env.q_and_qd_from_obs(obs_next)
        qdd = (qd_next - qd) / 0.01

        sys = env.sys
        x, xd = kinematics.forward(sys, q, qd)
        state = GeneralizedState.init(q, qd, x, xd)
        state = dynamics.transform_com(sys, state)
        state = mass.matrix_inv(sys, state, sys.matrix_inv_iterations)
        state = state.replace(constraint_forces=jp.zeros_like(state.con_diag))
        qf_smooth = dynamics.forward(sys, state, jp.zeros(sys.qd_size()))

        norm_obs = env._get_obs_approx(state, obs)
        torques, _ = env.torque_pd_control(scaled_action, norm_obs)
        tau = actuator.to_tau(sys, torques, state.q, state.qd)

        qf_smooth = (qf_smooth + tau)
        qf_constraint = state.mass_mx @ qdd - qf_smooth
        return qf_constraint

    reals = jax.vmap(est_constraint_forces)(norm_obs, norm_obs_next,
                                            scaled_action)

    with open(save_path, 'wb') as f:
        dill.dump((time, preds, reals), f)


def add_kwargs_to_fn(partial_fn, **kwargs):
    """add the kwargs to the passed in partial function"""
    for param in kwargs:
        partial_fn.keywords[param] = kwargs[param]
    return partial_fn


if __name__ == '__main__':
    validate_model()

from omegaconf import DictConfig, OmegaConf
import hydra
import os
import functools as ft
import dill
import wandb


@hydra.main(config_path="configs", config_name="go1")
def validate_model(cfg: DictConfig):
    train_model = True
    hardware_data = True

    if hardware_data:
        data_path = '/home/jl79444/dev/d4po/saved_policies/model_validation/hardware_data/ssrl_data.pkl'
        sac_ts_path = '/home/jl79444/dev/d4po/saved_policies/model_validation/hardware_data/sac_ts.pkl'
        model_path = None
    else:
        data_path = '/home/jl79444/dev/d4po/saved_policies/model_validation/sim_data/ssrl_state.pkl'
        model_path = '/home/jl79444/dev/d4po/saved_policies/model_validation/sim_data/training_state.pkl'

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
    from brax.training.agents.ssrl import losses as ssrl_losses
    from jax import numpy as jp
    import jax

    config_dict = OmegaConf.to_container(cfg, resolve=True,
                                         throw_on_missing=True)
    wandb.init(project=('go1_ssrl_model_validate'),
               entity=cfg.wandb.entity,
               config=config_dict)

    # create env fn
    env_dict = {'Go1GoFast': Go1GoFast}
    env_fn = ft.partial(env_dict[cfg.env], backend='generalized')
    env_kwargs = cfg.env_ssrl
    env_fn = add_kwargs_to_fn(env_fn, **env_kwargs)
    env = env_fn()

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

    if train_model:
        training_state, model_metrics = ssrl.train_model(
            ms.training_state, ms.env_buffer_state, ms.env_buffer,
            ms.constants, ms.local_key)
        # save training_state for debugging
        # with open('training_state.pkl', 'wb') as f:
        #     dill.dump(training_state, f)
    else:
        # load model
        with open(model_path, 'rb') as f:
            training_state = dill.load(f)
    ms = ms.replace(training_state=training_state)

    scale_fn = base.Scaler.transform
    model_network = model_network_factory(ms.constants.obs_size, output_dim)
    loss_constants = ms.constants.replace(
        model_loss_horizon=50
    )
    model_loss = ssrl_losses.make_losses(model_network, scale_fn,
                                           loss_constants,
                                           mean_loss_over_horizon=True)

    # Create a dataset using 400 random starting points
    env_buffer = ms.env_buffer
    c = loss_constants
    buffer_size = env_buffer.size(env_buffer_state)
    data = env_buffer_state.data[:buffer_size]
    all_transitions = env_buffer._unflatten_fn(data)
    dataset = base.Dataset(all_transitions, c.model_loss_horizon)
    idxs = jax.random.randint(ms.local_key, (400,),
                              minval=0, maxval=len(dataset))
    test_transitions = dataset[idxs]

    def duplicate_for_ensemble(transitions):
        num_samples = transitions.observation.shape[0]
        transitions = jax.tree_map(lambda x: jp.expand_dims(x, axis=0),
                                   transitions)
        transitions = jax.tree_map(
            lambda x: jp.broadcast_to(
                x, (c.model_ensemble_size, num_samples) + x.shape[2:]),
            transitions)
        return transitions
    test_transitions = duplicate_for_ensemble(test_transitions)

    # transpose to (num_batches, batch_size, horizon, ensemble_size, dim)
    test_transitions = jax.tree_map(lambda x: jp.swapaxes(x, 0, 1),
                                    test_transitions)
    test_transitions = jax.tree_map(lambda x: jp.swapaxes(x, 1, 2),
                                    test_transitions)

    # with jax.disable_jit():  # REMOVE
    params = ms.training_state.model_params
    model_params = params.pop('params')
    _, mean_loss = model_loss(model_params,
                              params,
                              ms.training_state.scaler_params,
                              test_transitions.observation,
                              test_transitions.next_observation,
                              test_transitions.action,
                              test_transitions.discount)

    for step, loss in enumerate(mean_loss):
        wandb.log({'mean_loss': loss}, step=step)


def add_kwargs_to_fn(partial_fn, **kwargs):
    """add the kwargs to the passed in partial function"""
    for param in kwargs:
        partial_fn.keywords[param] = kwargs[param]
    return partial_fn


if __name__ == '__main__':
    validate_model()

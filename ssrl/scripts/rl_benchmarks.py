import functools as ft
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from datetime import datetime


@hydra.main(config_path="configs", config_name="rl_benchmarks")
def train(cfg: DictConfig):

    # set environment variables before loading anything from jax/brax
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)

    from brax import envs
    from brax.training.agents.ssrl import train as ssrl
    from brax.training.agents.ssrl import networks as ssrl_networks

    env_configs = {
        'ant2': cfg.env_ant2,
        'hopper2': cfg.env_hopper2,
        'walker2d2': cfg.env_walker2d2,
    }

    get_env_fn = ft.partial(envs.get_environment, cfg.env)
    get_env_fn = add_kwargs_to_fn(get_env_fn, **env_configs[cfg.env])
    env = get_env_fn()

    def progress_fn(steps, metrics):
        metrics['steps'] = steps
        print(steps, metrics)
        if cfg.wandb.log:
            wandb.log(metrics)

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

    run_name = cfg.algo + '_' + datetime.now().strftime("%Y-%m-%d_%H%M_%S")
    if cfg.run_name is not None:
        run_name = cfg.run_name

    # start wandb
    if cfg.wandb.log:
        config_dict = OmegaConf.to_container(cfg, resolve=True,
                                             throw_on_missing=True)
        wandb.init(project=(cfg.env + '_' + cfg.algo),
                   entity=cfg.wandb.entity,
                   name=run_name,
                   config=config_dict)
        print(OmegaConf.to_yaml(cfg))

    if cfg.algo == 'ssrl':
        model_network_factory = ft.partial(
            ssrl_networks.make_model_network,
            probabilistic=cfg.ssrl_model_probabilistic)

        output_dim = env.sys.qd_size()
        if cfg.ssrl_dynamics_fn == 'mbpo':
            output_dim = env.observation_size
        dynamics_fns = {
            'contact': getattr(env, 'dynamics_contact', None),
            'contact_correct': getattr(env, 'dynamics_contact_correct', None),
            'contact_integrate_only': getattr(
                env, 'dynamics_contact_integrate_only', None),
            'integrate': getattr(env, 'dynamics_integrate', None),
            'mbpo': getattr(env, 'dynamics_mbpo', None),
            'exact': getattr(env, 'dynamics_exact', None),
            'all_forces': getattr(env, 'dynamics_all_forces', None),
            'all_forces_correct': getattr(env, 'dynamics_all_forces_correct',
                                          None),
        }
        if cfg.ssrl_dynamics_fn not in dynamics_fns:
            raise ValueError(f'Unknown dynamics_fn: {cfg.ssrl_dynamics_fn}')
        if dynamics_fns[cfg.ssrl_dynamics_fn] is None:
            raise ValueError(f'No dynamics_fn for {cfg.ssrl_dynamics_fn}')
        ssrl_fn = ft.partial(ssrl.train)
        ssrl_fn = add_kwargs_to_fn(ssrl_fn, **cfg.ssrl)
        ssrl_fn(
            environment=env,
            low_level_control_fn=env.low_level_control,
            dynamics_fn=dynamics_fns[cfg.ssrl_dynamics_fn],
            reward_fn=env.ssrl_reward_fn,
            model_output_dim=output_dim,
            model_horizon_fn=model_horizon_fn,
            hallucination_updates_per_training_step=hupts_fn,
            model_network_factory=model_network_factory,
            progress_fn=progress_fn)
    else:
        raise ValueError(f'Unknown algorithm: {cfg.algo}')


def add_kwargs_to_fn(partial_fn, **kwargs):
    """add the kwargs to the passed in partial function"""
    for param in kwargs:
        partial_fn.keywords[param] = kwargs[param]
    return partial_fn


if __name__ == '__main__':
    train()

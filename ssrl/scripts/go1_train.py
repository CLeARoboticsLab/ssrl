import os
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import functools as ft
from pathlib import Path
import hydra
import wandb
import dill


def int_multiply(x, y):
    return int(x * y)


OmegaConf.register_new_resolver("int_multiply", int_multiply)


@hydra.main(config_path="configs", config_name="go1")
def train_go1(cfg: DictConfig):

    # set environment variables before loading anything from jax/brax
    # os.environ["XLA_FLAGS"] = '--xla_gpu_deterministic_ops=true'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)

    # use 64-bit precision to limit rollout dynamics mismatch
    from jax import config
    config.update("jax_enable_x64", True)

    from brax.envs.go1_go_fast import Go1GoFast
    from brax.training.agents.sac2 import train as sac
    from brax.training.agents.ssrl import train as ssrl
    from brax.robots.go1 import networks as go1_networks
    from brax.evaluate import evaluate
    from brax.io import html
    import jax

    if cfg.algo == 'sac':
        use_wandb = cfg.wandb.log_sac
        env_kwargs = cfg.env_sac
        save_policy = cfg.save_policy.sac
        policy_name = 'go1_sac_policy.pkl'
        eplen = cfg.sac.episode_length
        deterministic_eval = cfg.sac.deterministic_eval
    if cfg.algo == 'ssrl':
        use_wandb = cfg.wandb.log_ssrl
        env_kwargs = cfg.env_ssrl
        save_policy = cfg.save_policy.ssrl
        policy_name = 'go1_ssrl_policy.pkl'
        eplen = cfg.ssrl.episode_length
        deterministic_eval = cfg.ssrl.deterministic_eval

    run_name = cfg.algo + '_' + datetime.now().strftime("%Y-%m-%d_%H%M_%S")
    if cfg.run_name is not None:
        run_name = cfg.run_name

    # start wandb
    if use_wandb:
        config_dict = OmegaConf.to_container(cfg, resolve=True,
                                             throw_on_missing=True)
        wandb.init(project=('go1_' + cfg.algo),
                   entity=cfg.wandb.entity,
                   name=run_name,
                   config=config_dict)
    print(OmegaConf.to_yaml(cfg))
    print(f"Running on GPU {os.environ['CUDA_VISIBLE_DEVICES']}")

    # create env fn
    env_dict = {'Go1GoFast': Go1GoFast}
    env_fn = ft.partial(env_dict[cfg.env], backend='generalized')
    env_fn = add_kwargs_to_fn(env_fn, **env_kwargs)
    env = env_fn()

    # progress functions
    best_reward = -float('inf')
    best_reward_domain_curr = -float('inf')
    best_params_domain_curr = None
    epoch = 0

    def progress(num_steps, metrics):
        metrics['steps'] = num_steps
        print("Steps / Eval: ", num_steps)
        if 'eval/episode_reward' in metrics:
            nonlocal best_reward
            print("Reward is ", metrics['eval/episode_reward'])
            best_reward = max(best_reward, metrics['eval/episode_reward'])
            metrics['eval/best_reward'] = best_reward
        if 'eval/episode_forward_vel' in metrics:
            metrics['eval/episode_forward_vel'] = (
                metrics['eval/episode_forward_vel']
                / (eplen / cfg.common.action_repeat)
            )
        if use_wandb and cfg.num_seeds == 1:
            wandb.log(metrics)

    def policy_params_fn(current_step, make_policy, params, metrics):
        nonlocal epoch

        # store the best policy when using domain curriculum
        nonlocal best_reward_domain_curr
        nonlocal best_params_domain_curr

        # save policies at each evaluation step
        if ((cfg.save_policy.sac_all and cfg.algo == 'sac')
            or (cfg.save_policy.ssrl_all and cfg.algo == 'ssrl')):
            if cfg.sweep_name is None:
                fname = f"{cfg.algo}_{run_name}_step_{current_step:.0f}_rew_{metrics['eval/episode_reward']:.0f}.pkl"
                path = (Path(__file__).parent.parent
                        / 'saved_policies'
                        / fname)
            else:
                fname = f"env{cfg.env}_{cfg.env_common.reward_type}_zvelchng{cfg.env_common.z_vel_change_rew_weight}_rew_{metrics['eval/episode_reward']:06.0f}_epoch_{epoch}.pkl"
                path = (Path(__file__).parent.parent
                        / 'saved_policies'
                        / cfg.algo
                        / cfg.sweep_name
                        / fname)
            Path(path.parent).mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                dill.dump(params, f)

        # render evals
        if (use_wandb and cfg.render_during_training
                and epoch % cfg.render_epoch_interval == 0):
            key = jax.random.PRNGKey(cfg.render_seed)
            eval_results = evaluate(
                params=params,
                env_unwrapped=env,
                make_policy=make_policy,
                episode_length=eplen,
                action_repeat=cfg.common.action_repeat,
                key=key,
                obs_history_length=cfg.common.obs_history_length,
                deterministic=deterministic_eval,
                jit=True,
            )
            pipeline_states = eval_results[1]
            render_html = html.render(env.sys.replace(dt=env.dt),
                                      pipeline_states,
                                      height=500)
            wandb.log(
                {f"Render at step {current_step}": wandb.Html(render_html)})

        epoch += 1

    saved_policies_dir = Path(__file__).parent.parent / 'saved_policies'

    if cfg.algo == 'sac':
        # make networks
        network_factory = go1_networks.sac_network_factory(cfg)

        # perform training
        train_fn = ft.partial(sac.train)
        train_fn = add_kwargs_to_fn(train_fn, **cfg.sac)
        make_policy, params, metrics = train_fn(
            environment=env,
            network_factory=network_factory,
            progress_fn=progress,
            policy_params_fn=policy_params_fn
        )

    if cfg.algo == 'ssrl':
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

        # preload sac policy
        sac_ts = None
        if cfg.ssrl_start_with_sac:
            params, make_policy, sac_ts = go1_networks.make_sac_networks(
                cfg, env, saved_policies_dir)

        # perform training
        train_fn = ft.partial(ssrl.train)
        train_fn = add_kwargs_to_fn(train_fn, **cfg.ssrl)
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
        state = train_fn(
            environment=env,
            low_level_control_fn=env.low_level_control,
            dynamics_fn=dynamics_fn,
            reward_fn=env.ssrl_reward_fn,
            model_output_dim=output_dim,
            model_horizon_fn=model_horizon_fn,
            hallucination_updates_per_training_step=hupts_fn,
            sac_training_state=sac_ts,
            model_network_factory=model_network_factory,
            sac_network_factory=sac_network_factory,
            progress_fn=progress,
            policy_params_fn=policy_params_fn
        )
        with open(run_name + 'ssrl_state.pkl', 'wb') as f:
            dill.dump(state, f)

    # save policy
    if save_policy:
        path = (Path(__file__).parent.parent
                / 'saved_policies'
                / policy_name)
        with open(path, 'wb') as f:
            dill.dump(params, f)


def add_kwargs_to_fn(partial_fn, **kwargs):
    """add the kwargs to the passed in partial function"""
    for param in kwargs:
        partial_fn.keywords[param] = kwargs[param]
    return partial_fn


def dict_mean(dict_list):
    """Take a list of dicts with the same keys and return a dict with the mean
    of each key"""
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def lin_interp(start, end, i, n):
    """Linear interpolation between start and end for zero-indexed i out of n
    total iterations"""
    return start + i / (n-1) * (end - start)


if __name__ == '__main__':
    train_go1()

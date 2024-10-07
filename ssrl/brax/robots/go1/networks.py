from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac2 import networks as sac_networks
from brax.training.agents.ssrl import networks as ssrl_networks
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.envs.base import RlwamEnv

from omegaconf import DictConfig
from flax import linen
from pathlib import Path
import dill
import functools as ft
import jax
import jax.numpy as jp

_activations = {
    'swish': linen.swish,
    'tanh': linen.tanh
}


def sac_network_factory(cfg: DictConfig):
    activation = _activations[cfg.actor_network.activation]

    # network factory
    network_factory = ft.partial(
        sac_networks.make_sac_networks,
        hidden_layer_sizes=((cfg.actor_network.hidden_size,)
                            * cfg.actor_network.hidden_layers),
        activation=activation
    )
    return network_factory


def ssrl_network_factories(cfg: DictConfig):
    activation = _activations[cfg.actor_network.activation]

    # network factory
    sac_network_factory = ft.partial(
        sac_networks.make_sac_networks,
        hidden_layer_sizes=((cfg.actor_network.hidden_size,)
                            * cfg.actor_network.hidden_layers),
        activation=activation,
        policy_max_std=cfg.actor_network.max_std
    )
    model_network_factory = ft.partial(
        ssrl_networks.make_model_network,
        hidden_size=cfg.ssrl_model.hidden_size,
        ensemble_size=cfg.ssrl_model.ensemble_size,
        num_elites=cfg.ssrl_model.num_elites,
        probabilistic=cfg.ssrl_model.probabilistic)
    return sac_network_factory, model_network_factory


def make_ppo_networks(cfg: DictConfig, saved_policies_dir: Path,
                      env: RlwamEnv, ppo_params_path: Path = None):
    if ppo_params_path is not None:
        path = ppo_params_path
    else:
        path = saved_policies_dir / 'go1_ppo_policy.pkl'
    with open(path, 'rb') as f:
        params = dill.load(f)

    # create the policy network
    normalize = lambda x, y: x
    if cfg.common.normalize_observations:
        normalize = running_statistics.normalize
    ppo_network = ppo_networks.make_ppo_networks(
        env.observation_size*cfg.contact_generate.obs_history_length,
        env.action_size,
        preprocess_observations_fn=normalize,
        policy_hidden_layer_sizes=((cfg.actor_network.hidden_size,)
                                   * cfg.actor_network.hidden_layers)
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    return params, make_policy


def make_sac_networks(cfg: DictConfig, env: RlwamEnv,
                      saved_policies_dir: Path = None,
                      sac_ts_path: Path = None,
                      seed: int = 0):
    # create the policy network
    activation = _activations[cfg.actor_network.activation]
    normalize = lambda x, y: x
    if cfg.common.normalize_observations:
        normalize = running_statistics.normalize
    sac_network = sac_networks.make_sac_networks(
        env.observation_size*cfg.common.obs_history_length,
        env.action_size,
        preprocess_observations_fn=normalize,
        hidden_layer_sizes=((cfg.actor_network.hidden_size,)
                            * cfg.actor_network.hidden_layers),
        activation=activation,
        policy_max_std=cfg.actor_network.max_std
    )
    make_policy = sac_networks.make_inference_fn(sac_network)
    
    # load or initialize params
    if not (saved_policies_dir is None and sac_ts_path is None):
        if sac_ts_path is not None:
            path = sac_ts_path
        elif saved_policies_dir is not None:
            path = saved_policies_dir / 'go1_sac_policy.pkl'
        with open(path, 'rb') as f:
            sac_ts = dill.load(f)
        params = (sac_ts.normalizer_params, sac_ts.policy_params)
    else:
        normalizer_params = running_statistics.init_state(
            specs.Array((env.observation_size*cfg.common.obs_history_length,),
                        jp.float32))
        key = jax.random.PRNGKey(seed)
        policy_params = sac_network.policy_network.init(key)
        params = (normalizer_params, policy_params)
        sac_ts = None

    return params, make_policy, sac_ts


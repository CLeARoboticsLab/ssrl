from brax.training import types
from brax.training.agents.ssrl import base
from typing import Tuple, Dict
from flax import linen as nn
from flax import struct
from jax import numpy as jp
import jax

ModelParams = Tuple[base.ScalerParams, types.Params]


class EnsembleDense(nn.Module):
    """Ensemble Dense module.
    (ensemble_size, input_size) -> (ensemble_size, features)
    """
    features: int
    ensemble_size: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x):
        kernel = self.param('kernel', self.kernel_init,
                            (self.ensemble_size, x.shape[-1], self.features))
        bias = self.param('bias', self.bias_init,
                          (self.ensemble_size, self.features))
        return jax.vmap(jp.matmul)(x, kernel) + bias


class EnsembleModel(nn.Module):
    """Ensemble model.
    (ensemble_size, obs_size + action_size) -> (ensemble_size, output_dim)
        -> (means: (ensemble_size, output_dim)
            logvars: (ensemble_size, obs_size)
    """
    obs_size: int
    output_dim: int
    ensemble_size: int
    num_elites: int
    hidden_size: int
    probabilistic: bool
    weight_decays: Dict[str, jp.ndarray] = struct.field(
        default_factory=lambda: {
            'ed1': 0.000025,
            'ed2': 0.00005,
            'ed3': 0.000075,
            'ed4': 0.000075,
            'ed5': 0.0001,
        }
    )

    def setup(self):
        self.max_logvar = 0.5 * jp.ones((1, self.obs_size))
        self.min_logvar = -10. * jp.ones((1, self.obs_size))
        self.act = nn.swish

        self.ed1 = EnsembleDense(self.hidden_size, self.ensemble_size)
        self.ed2 = EnsembleDense(self.hidden_size, self.ensemble_size)
        self.ed3 = EnsembleDense(self.hidden_size, self.ensemble_size)
        self.ed4 = EnsembleDense(self.hidden_size, self.ensemble_size)
        out_dim = (self.output_dim + self.obs_size if self.probabilistic
                   else self.output_dim)
        self.ed5 = EnsembleDense(out_dim, self.ensemble_size)

    @nn.compact
    def __call__(self, x):
        elite_idxs = self.variable('elites', 'idxs',  # noqa: F841
                                   lambda: jp.arange(self.num_elites))

        x = self.ed1(x)
        x = self.act(x)
        x = self.ed2(x)
        x = self.act(x)
        x = self.ed3(x)
        x = self.act(x)
        x = self.ed4(x)
        x = self.act(x)
        x = self.ed5(x)

        if self.probabilistic:
            mean, logvar = jp.split(x, [self.output_dim], axis=-1)
            logvar = self.max_logvar - nn.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + nn.softplus(logvar - self.min_logvar)
            return mean, logvar
        else:
            return x, jp.zeros((self.ensemble_size, self.obs_size))


def make_inference_fn(
    ensemble_model: EnsembleModel,
    preprocess_fn: lambda x, y: (x, y),
    c: base.Constants
):

    def make_model(model_params: ModelParams):

        def model(obs_stack: types.Observation, action: types.Action,
                  key: types.PRNGKey):
            scaler_params, params = model_params
            obs = obs_stack[:c.obs_size]
            proc_obs, proc_act = preprocess_fn(obs_stack, action,
                                               scaler_params)
            x = jp.concatenate([proc_obs, proc_act], axis=-1)
            # repeat x across ensemble dimension
            x = jp.tile(x, (ensemble_model.ensemble_size,) + (1,))

            means, logvars = ensemble_model.apply(params, x)
            key_normal, key_choice = jax.random.split(key)

            # means has shape (ensemble_size, output_dim)
            # randomly select a mean and std from the elites
            idx = jax.random.choice(key_choice, params['elites']['idxs'])
            mean = means[idx]
            logvar = logvars[idx]
            std = jp.sqrt(jp.exp(logvar))
            if not c.model_probabilistic:
                std = jp.zeros_like(std)

            # propagate the mean through the dynamics
            def f(carry, in_element_unused):
                obs = carry
                u = c.low_level_control_fn(action, obs)
                obs_next = c.dynamics_fn(obs, u, mean)
                return obs_next, u
            obs_next_mean, us = jax.lax.scan(
                f, obs, (), length=c.policy_repeat)

            # add noise to the propagated mean
            obs_next = obs_next_mean + std * jax.random.normal(
                key_normal, obs_next_mean.shape)

            # compute reward
            reward = c.reward_fn(obs, obs_next, jp.mean(us), action)

            return obs_next, reward

        return model

    return make_model


def make_model_network(obs_size: int,
                       output_dim: int,
                       hidden_size: int = 200,
                       ensemble_size: int = 7,
                       num_elites: int = 5,
                       probabilistic: bool = False):
    """Creates a model network."""
    assert ensemble_size >= num_elites
    return EnsembleModel(obs_size, output_dim, ensemble_size, num_elites,
                         hidden_size, probabilistic)

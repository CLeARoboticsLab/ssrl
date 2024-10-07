from brax import envs
from brax.training import acting
from brax.training import replay_buffers
from brax.training.types import Params, PRNGKey, Observation, Action
from brax.training.types import Transition
from brax.training.agents.sac2 import base as sac_base

from jax import numpy as jp
from typing import Any, Callable, Protocol, Tuple
import flax
import optax
import jax


ReplayBufferState = Any
Reward = jp.ndarray
done_fn = Callable[[jp.ndarray], jp.ndarray]
reward_fn = Callable[
    [jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray], jp.ndarray]
low_level_control_fn = Callable[[jp.ndarray, jp.ndarray], jp.ndarray]
dynamics_fn = Callable[[jp.ndarray, jp.ndarray, jp.ndarray], jp.ndarray]


class Model(Protocol):

    def __call__(
        self,
        observation: Observation,
        action: Action,
        key: PRNGKey,
    ) -> Tuple[Observation, Reward]:
        pass


class ModelEnv(envs.Env):
    """Environment for the model which hallucinates transitions."""
    def __init__(self, done_fn: Callable, observation_size: int,
                 action_size: int):
        self._done_fn = done_fn
        self._observation_size = observation_size
        self._action_size = action_size

    def reset(self, rng: jp.ndarray):
        return

    def step(self, state: envs.State, action: jp.ndarray) -> envs.State:
        next_obs = state.info['next_obs']
        reward = state.info['reward']
        done = self._done_fn(next_obs)
        nstate = state.replace(obs=next_obs, reward=reward, done=done)
        return nstate

    @property
    def observation_size(self):
        return self._observation_size

    @property
    def action_size(self):
        return self._action_size

    @property
    def backend(self):
        return None


class Dataset:
    def __init__(self, transitions: Transition, horizon: int):
        self._transitions = transitions
        self._h = horizon

    def __len__(self):
        return self._transitions.observation.shape[0] - self._h

    def __getitem__(self, idxs: jp.ndarray):
        def f(idx):
            transitions = jax.tree_util.tree_map(
                lambda x: jax.lax.dynamic_slice_in_dim(
                    x, idx, self._h),
                self._transitions)
            return transitions
        return jax.vmap(f)(idxs)


@flax.struct.dataclass
class ScalerParams:
    obs_mu: jp.ndarray
    obs_std: jp.ndarray
    act_mu: jp.ndarray
    act_std: jp.ndarray


class Scaler:
    @staticmethod
    def init(obs_size: int, act_size: int):
        return ScalerParams(
            obs_mu=jp.zeros(obs_size), obs_std=jp.ones(obs_size),
            act_mu=jp.zeros(act_size), act_std=jp.ones(act_size))

    @staticmethod
    def fit(obs: jp.ndarray, act: jp.ndarray):
        obs_mu = jp.mean(obs, axis=0)
        obs_std = jp.std(obs, axis=0) + 1e-6
        act_mu = jp.mean(act, axis=0)
        act_std = jp.std(act, axis=0) + 1e-6
        return ScalerParams(obs_mu=obs_mu, obs_std=obs_std,
                            act_mu=act_mu, act_std=act_std)

    @staticmethod
    def transform(obs: jp.ndarray, act: jp.ndarray, params: ScalerParams):
        obs = (obs - params.obs_mu) / params.obs_std
        act = (act - params.act_mu) / params.act_std
        return obs, act

    @staticmethod
    def inverse_transform(obs: jp.ndarray, act: jp.ndarray,
                          params: ScalerParams):
        obs = obs * params.obs_std + params.obs_mu
        act = act * params.act_std + params.act_mu
        return obs, act


@flax.struct.dataclass
class Constants:
    num_epochs: int
    model_trains_per_epoch: int
    training_steps_per_model_train: int
    env_steps_per_training_step: int
    hallucination_updates_per_training_step_fn: Callable
    model_rollouts_per_hallucination_update: int
    sac_grad_updates_per_hallucination_update: int
    env_steps_per_epoch: int
    init_exploration_steps: int
    clear_model_buffer_after_model_train: bool
    num_envs: int
    obs_size: int
    obs_hist_len: int
    action_size: int
    action_repeat: int
    model_learning_rate: float
    model_training_batch_size: int
    model_training_max_sgd_steps_per_epoch: int
    model_training_max_epochs: int
    model_training_convergence_criteria: float
    model_training_consec_converged_epochs: int
    model_training_abs_criteria: float
    model_training_test_ratio: float
    model_training_weight_decay: bool
    model_training_stop_gradient: bool
    model_loss_horizon: int
    model_horizon_fn: Callable
    model_ensemble_size: int
    model_num_elites: int
    model_probabilistic: bool
    model_max_train_batches: int
    model_max_test_batches: int
    make_model: Callable
    model_loss: Callable
    sac_batch_size: int
    real_ratio: float
    model_update: Callable
    low_level_control_fn: low_level_control_fn
    dynamics_fn: dynamics_fn
    reward_fn: reward_fn
    policy_repeat: int
    max_model_buffer_size: int
    deterministic_in_env: bool
    make_policy_env: Callable


@flax.struct.dataclass
class TrainingState:
    model_optimizer_state: optax.OptState
    model_params: Params
    scaler_params: ScalerParams
    env_steps: jp.ndarray


@flax.struct.dataclass
class MbpoState:
    training_state: TrainingState
    sac_state: sac_base.SacState
    constants: Constants
    env: envs.Env
    env_buffer: replay_buffers.UniformSamplingQueue
    env_buffer_state: ReplayBufferState
    model_env: ModelEnv
    evaluator: acting.Evaluator
    local_key: PRNGKey
    model_horizon: int
    hallucination_updates_per_training_step: int

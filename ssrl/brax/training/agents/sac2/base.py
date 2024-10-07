from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.training.acme import running_statistics
from brax.training import acting
from brax.training import replay_buffers
from brax import envs

from typing import Callable, Any
import flax
import optax
import jax.numpy as jnp

ReplayBufferState = Any


@flax.struct.dataclass
class TrainingState:
    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: running_statistics.RunningStatisticsState


@flax.struct.dataclass
class Constants:
    num_envs: int
    action_repeat: int
    make_policy: Callable
    reward_scaling: float
    discounting: float
    tau: float
    fixed_alpha: float
    num_training_steps_per_epoch: int
    num_prefill_actor_steps: int
    env_steps_per_actor_step: int
    grad_updates_per_step: int
    action_size: int
    alpha_update: Callable
    critic_update: Callable
    actor_update: Callable


@flax.struct.dataclass
class SacState:
    training_state: TrainingState
    constants: Constants
    env: envs.Env
    replay_buffer: replay_buffers.UniformSamplingQueue
    buffer_state: ReplayBufferState
    evaluator: acting.Evaluator
    local_key: PRNGKey

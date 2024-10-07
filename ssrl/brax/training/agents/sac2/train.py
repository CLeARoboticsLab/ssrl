"""Soft Actor-Critic training.

The same as the original Brax SAC implementation, but with functions separated,
so that they can be called individually by other algorithms (such as MBPO).
"""

import functools
import time
from typing import Callable, Optional, Tuple, Union

from absl import logging
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import replay_buffers
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.sac2 import base
from brax.training.agents.sac2 import losses as sac_losses
from brax.training.agents.sac2 import networks as sac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
import jax
import jax.numpy as jnp
import optax

Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]


def train(
    environment: Union[envs_v1.Env, envs.Env],
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    obs_history_length: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = False,
    reward_scaling: float = 1.,
    tau: float = 0.005,
    fixed_alpha: Optional[float] = None,  # if supplied, alpha will be fixed to this value  # noqa: E501
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        sac_networks.SACNetworks] = sac_networks.make_sac_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None
):
    ss = initizalize_training(
        environment=environment,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        action_repeat=action_repeat,
        obs_history_length=obs_history_length,
        num_envs=num_envs,
        num_eval_envs=num_eval_envs,
        learning_rate=learning_rate,
        discounting=discounting,
        seed=seed,
        batch_size=batch_size,
        num_evals=num_evals,
        normalize_observations=normalize_observations,
        reward_scaling=reward_scaling,
        tau=tau,
        fixed_alpha=fixed_alpha,
        min_replay_size=min_replay_size,
        max_replay_size=max_replay_size,
        grad_updates_per_step=grad_updates_per_step,
        deterministic_eval=deterministic_eval,
        network_factory=network_factory,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        eval_env=eval_env
    )

    # Run initial eval
    metrics = {}
    if num_evals > 1:
        metrics = ss.evaluator.run_evaluation(
            (ss.training_state.normalizer_params,
             ss.training_state.policy_params),
            training_metrics={})
        logging.info(metrics)
        progress_fn(0, metrics)
        policy_params_fn(0, ss.constants.make_policy,
                         (ss.training_state.normalizer_params,
                          ss.training_state.policy_params),
                         metrics)

    # Prefill the replay buffer.
    def prefill_replay_buffer(
        training_state: base.TrainingState, env_state: envs.State,
        buffer_state: base.ReplayBufferState, key: PRNGKey
    ) -> Tuple[base.TrainingState, envs.State,
               base.ReplayBufferState, PRNGKey]:

        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            new_normalizer_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params,
                training_state.policy_params,
                env_state, buffer_state, key, ss.constants, ss.env,
                ss.replay_buffer)
            new_training_state = training_state.replace(
                normalizer_params=new_normalizer_params,
                env_steps=(training_state.env_steps
                           + ss.constants.env_steps_per_actor_step))
            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f, (training_state, env_state, buffer_state, key), (),
            length=ss.constants.num_prefill_actor_steps)[0]

    local_key, env_key, prefill_key,  = jax.random.split(ss.local_key, 3)
    env_keys = jax.random.split(env_key, num_envs)
    env_state = ss.env.reset(env_keys)
    ss = ss.replace(local_key=local_key)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        ss.training_state, env_state, ss.buffer_state, prefill_key)
    ss = ss.replace(training_state=training_state,
                    buffer_state=buffer_state)
    replay_size = ss.replay_buffer.size(buffer_state)
    logging.info('replay size after prefill %s', replay_size)
    assert replay_size >= min_replay_size

    num_evals_after_init = max(num_evals - 1, 1)
    current_step = 0
    training_walltime = 0
    for _ in range(num_evals_after_init):
        logging.info('step %s', current_step)
        # optimization

        (ss, training_metrics, training_walltime,
         env_state) = sim_training_epoch_with_timing(
            ss, training_walltime, env_state)

        current_step = ss.training_state.env_steps
        # Run evals.
        metrics = ss.evaluator.run_evaluation(
            (ss.training_state.normalizer_params,
             ss.training_state.policy_params),
            training_metrics)
        logging.info(metrics)
        progress_fn(current_step, metrics)
        policy_params_fn(current_step, ss.constants.make_policy,
                         (ss.training_state.normalizer_params,
                          ss.training_state.policy_params),
                         metrics)

    trained_params = (ss.training_state.normalizer_params,
                      ss.training_state.policy_params)

    return ss.constants.make_policy, trained_params, metrics


def initizalize_training(
    environment: Union[envs_v1.Env, envs.Env],
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    obs_history_length: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = False,
    reward_scaling: float = 1.,
    tau: float = 0.005,
    fixed_alpha: Optional[float] = None,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        sac_networks.SACNetworks] = sac_networks.make_sac_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None
) -> base.SacState:

    if max_replay_size is None:
        max_replay_size = num_timesteps

    # The number of environment steps executed for every `actor_step()` call.
    env_steps_per_actor_step = action_repeat * num_envs
    # equals to ceil(min_replay_size / env_steps_per_actor_step)
    num_prefill_actor_steps = -(-min_replay_size // num_envs)
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert num_timesteps - num_prefill_env_steps >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of run_one_sac_epoch calls per run_sac_training.
    # equals to
    # ceil(num_timesteps - num_prefill_env_steps /
    #      (num_evals_after_init * env_steps_per_actor_step))
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps) //
        (num_evals_after_init * env_steps_per_actor_step))

    env = environment
    if isinstance(env, envs.Env):
        wrap_for_training = envs.training.wrap
    else:
        wrap_for_training = envs_v1.wrappers.wrap_for_training

    env = wrap_for_training(
        env, episode_length=episode_length, action_repeat=action_repeat,
        obs_history_length=obs_history_length)

    obs_size = env.observation_size*obs_history_length
    action_size = env.action_size

    normalize_fn = lambda x, y: x  # noqa: E731
    if normalize_observations:
        normalize_fn = running_statistics.normalize
    sac_network = network_factory(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn)
    make_policy = sac_networks.make_inference_fn(sac_network)

    alpha_optimizer = optax.adam(learning_rate=3e-4)
    policy_optimizer = optax.adam(learning_rate=learning_rate)
    q_optimizer = optax.adam(learning_rate=learning_rate)

    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=0.,
        discount=0.,
        next_observation=dummy_obs,
        extras={
            'state_extras': {
                'truncation': 0.
            },
            'policy_extras': {}
        }
    )
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size * grad_updates_per_step)

    alpha_loss, critic_loss, actor_loss = sac_losses.make_losses(
        sac_network=sac_network,
        reward_scaling=reward_scaling,
        discounting=discounting,
        action_size=action_size)
    alpha_update = gradients.gradient_update_fn(
        alpha_loss, alpha_optimizer, pmap_axis_name=None)
    critic_update = gradients.gradient_update_fn(
        critic_loss, q_optimizer, pmap_axis_name=None)
    actor_update = gradients.gradient_update_fn(
        actor_loss, policy_optimizer, pmap_axis_name=None)

    global_key, local_key = jax.random.split(jax.random.PRNGKey(seed))

    # Training state init
    training_state = _init_training_state(
        key=global_key,
        obs_size=obs_size,
        sac_network=sac_network,
        alpha_optimizer=alpha_optimizer,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer)

    del global_key
    local_key, rb_key, eval_key = jax.random.split(local_key, 3)
    buffer_state = replay_buffer.init(rb_key)

    if not eval_env:
        eval_env = env
    else:
        eval_env = wrap_for_training(
            eval_env, episode_length=episode_length,
            action_repeat=action_repeat,
            obs_history_length=obs_history_length)

    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key)

    constants = base.Constants(
        num_envs=num_envs,
        action_repeat=action_repeat,
        make_policy=make_policy,
        reward_scaling=reward_scaling,
        discounting=discounting,
        tau=tau,
        fixed_alpha=fixed_alpha,
        num_training_steps_per_epoch=num_training_steps_per_epoch,
        num_prefill_actor_steps=num_prefill_actor_steps,
        env_steps_per_actor_step=env_steps_per_actor_step,
        grad_updates_per_step=grad_updates_per_step,
        action_size=action_size,
        alpha_update=alpha_update,
        critic_update=critic_update,
        actor_update=actor_update
    )

    sac_state = base.SacState(
        training_state=training_state,
        constants=constants,
        env=env,
        replay_buffer=replay_buffer,
        buffer_state=buffer_state,
        evaluator=evaluator,
        local_key=local_key
    )

    return sac_state


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    sac_network: sac_networks.SACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation
) -> base.TrainingState:
    """Inits the training state"""
    key_policy, key_q = jax.random.split(key)
    log_alpha = jnp.asarray(0., dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = sac_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    q_params = sac_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.float32))

    training_state = base.TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        normalizer_params=normalizer_params
    )
    return training_state


def get_experience(
    normalizer_params: running_statistics.RunningStatisticsState,
    policy_params: Params,
    env_state: envs.State,
    buffer_state: base.ReplayBufferState,
    key: PRNGKey,
    c: base.Constants,
    env: envs.Env,
    replay_buffer: replay_buffers.UniformSamplingQueue
) -> Tuple[running_statistics.RunningStatisticsState,
           Union[envs.State, envs_v1.State], base.ReplayBufferState]:
    make_policy = c.make_policy
    policy = make_policy((normalizer_params, policy_params))
    env_state, transitions = acting.actor_step(
        env, env_state, policy, key, extra_fields=('truncation',))

    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation)

    buffer_state = replay_buffer.insert(buffer_state, transitions)
    return normalizer_params, env_state, buffer_state


# Note that this is NOT a pure jittable method.
def sim_training_epoch_with_timing(
    ss: base.SacState,
    training_walltime: float,
    env_state: Union[envs.State, envs_v1.State]
):
    t = time.time()

    epoch_key, new_local_key = jax.random.split(ss.local_key)
    ss = ss.replace(local_key=new_local_key)

    (training_state, env_state, buffer_state,
     metrics) = sim_training_epoch(ss.training_state, env_state,
                                   ss.buffer_state, epoch_key,
                                   ss.constants, ss.env, ss.replay_buffer)

    ss = ss.replace(training_state=training_state, buffer_state=buffer_state)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (ss.constants.env_steps_per_actor_step *
           ss.constants.num_training_steps_per_epoch) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()}
    }
    return ss, metrics, training_walltime, env_state


@functools.partial(jax.jit, static_argnames=['c', 'env', 'rb'])
def sim_training_epoch(
        training_state: base.TrainingState,
        env_state: envs.State,
        buffer_state: base.ReplayBufferState,
        key: PRNGKey,
        c: base.Constants,
        env: envs.Env,
        rb: replay_buffers.UniformSamplingQueue
) -> Tuple[base.TrainingState, envs.State, base.ReplayBufferState, Metrics]:

    def f(carry, unused_t):
        ts, es, bs, k = carry
        k, new_key = jax.random.split(k)
        ts, es, bs, metrics = sim_training_step(ts, es, bs, k, c, env, rb)
        return (ts, es, bs, new_key), metrics

    (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
        f, (training_state, env_state, buffer_state, key), (),
        length=c.num_training_steps_per_epoch)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)

    return training_state, env_state, buffer_state, metrics


def sim_training_step(
    training_state: base.TrainingState,
    env_state: envs.State,
    buffer_state: base.ReplayBufferState,
    key: PRNGKey,
    c: base.Constants,
    env: envs.Env,
    replay_buffer: replay_buffers.UniformSamplingQueue
) -> Tuple[base.TrainingState, envs.State, base.ReplayBufferState, Metrics]:

    # get experience
    experience_key, training_key = jax.random.split(key)
    normalizer_params, env_state, buffer_state = get_experience(
        training_state.normalizer_params, training_state.policy_params,
        env_state, buffer_state, experience_key, c, env, replay_buffer)
    training_state = training_state.replace(
        normalizer_params=normalizer_params,
        env_steps=training_state.env_steps + c.env_steps_per_actor_step)

    # policy update
    training_state, buffer_state, metrics, _ = policy_update(
        replay_buffer, buffer_state, training_state, training_key, c)

    return training_state, env_state, buffer_state, metrics


@functools.partial(jax.jit, static_argnames=['c', 'replay_buffer',
                                             'external_buffer',
                                             'external_buffer_ratio'])
def policy_update(
        replay_buffer: replay_buffers.UniformSamplingQueue,
        buffer_state: base.ReplayBufferState,
        training_state: base.TrainingState,
        training_key: PRNGKey,
        c: base.Constants,
        external_buffer: Optional[replay_buffers.UniformSamplingQueue] = None,
        external_buffer_state: base.ReplayBufferState = {},
        external_buffer_ratio: float = 0.0
) -> Tuple[base.TrainingState, base.ReplayBufferState, Metrics]:
    """Update the policy by doing grad_updates_per_step sgd_steps. If an
    external buffer is provided, external_buffer_ratio*batch_size samples are
    mixed into the batch."""

    def sgd_step(
        carry: Tuple[base.TrainingState, PRNGKey],
        transitions: Transition
    ) -> Tuple[Tuple[base.TrainingState, PRNGKey], Metrics]:
        training_state, key = carry

        key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

        alpha_loss, alpha_params, alpha_optimizer_state = c.alpha_update(
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state)
        alpha = jnp.exp(training_state.alpha_params)
        if c.fixed_alpha:
            alpha = c.fixed_alpha
            alpha_params = jnp.log(alpha).astype(jnp.float32)
        critic_loss, q_params, q_optimizer_state = c.critic_update(
            training_state.q_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_q_params,
            alpha,
            transitions,
            key_critic,
            optimizer_state=training_state.q_optimizer_state)
        actor_loss, policy_params, policy_optimizer_state = c.actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.q_params,
            alpha,
            transitions,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state)

        new_target_q_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - c.tau) + y * c.tau,
            training_state.target_q_params, q_params)

        metrics = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': jnp.exp(alpha_params),
        }

        new_training_state = base.TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=new_target_q_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            normalizer_params=training_state.normalizer_params)
        return (new_training_state, key), metrics

    # sample buffer
    buffer_state, transitions = replay_buffer.sample(buffer_state)

    if external_buffer is not None:
        # sample external buffer
        batch_size = transitions.observation.shape[0]
        ext_size = int(batch_size * external_buffer_ratio
                       // c.grad_updates_per_step * c.grad_updates_per_step)
        external_buffer_state, external_transitions = external_buffer.sample(
            external_buffer_state)
        external_transitions = jax.tree_util.tree_map(
            lambda x: x[:ext_size],
            external_transitions)
        transitions = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            transitions, external_transitions)

        # shuffle external transitions into the batch
        training_key, key_shuf = jax.random.split(training_key)
        permuted_idxs = jax.random.permutation(key_shuf, batch_size + ext_size)
        transitions = jax.tree_map(lambda x: x[permuted_idxs], transitions)

    # Change the front dimension of transitions so 'update_step' is called
    # grad_updates_per_step times by the scan.
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (c.grad_updates_per_step, -1) + x.shape[1:]),
        transitions)

    # take sgd steps
    (training_state, _), metrics = jax.lax.scan(sgd_step,
                                                (training_state, training_key),
                                                transitions)

    metrics['buffer_current_size'] = replay_buffer.size(buffer_state)

    return training_state, buffer_state, metrics, external_buffer_state

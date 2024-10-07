"""Physics Informed Model-Based Policy Optimization.
"""

from brax import envs
from brax.training import types
from brax.training import acting
from brax.training import replay_buffers
from brax.training import gradients
from brax.training.types import Params, Policy
from brax.training.types import PRNGKey
from brax.training.types import Transition
from brax.training.acme import running_statistics
from brax.training.agents.ssrl import networks as ssrl_networks
from brax.training.agents.ssrl import base
from brax.training.agents.ssrl import losses as ssrl_losses
from brax.training.agents.sac2 import train as sac
from brax.training.agents.sac2 import networks as sac_networks
from brax.training.agents.sac2 import base as sac_base

from absl import logging
from typing import Callable, Optional, Tuple, Sequence, Union
import optax
from jax import numpy as jp
import jax
import functools
import math
import time

Metrics = types.Metrics


def train(
    environment: envs.Env,
    low_level_control_fn: base.low_level_control_fn,
    dynamics_fn: base.dynamics_fn,
    reward_fn: base.reward_fn,
    model_output_dim: int,
    episode_length: int,
    policy_repeat: int = 1,
    num_timesteps: Optional[int] = None,
    num_epochs: Optional[int] = None,
    model_trains_per_epoch: int = 4,
    training_steps_per_model_train: int = 250,
    env_steps_per_training_step: int = 1,
    hallucination_updates_per_training_step: Union[int, Callable] = 1,
    model_rollouts_per_hallucination_update: int = 400,
    sac_grad_updates_per_hallucination_update: int = 20,
    init_exploration_steps: int = 5000,
    clear_model_buffer_after_model_train: bool = True,
    action_repeat: int = 1,
    obs_history_length: int = 1,
    num_envs: int = 1,  # TODO remove num_envs as option or freeze at 1
    num_evals: int = 1,
    num_eval_envs: int = 1,
    policy_normalize_observations: bool = False,
    model_learning_rate: float = 1e-3,
    model_training_batch_size: int = 256,
    model_training_max_sgd_steps_per_epoch: Optional[int] = None,
    model_training_max_epochs: int = 1000,
    model_training_convergence_criteria: float = 0.01,
    model_training_consec_converged_epochs: int = 6,
    model_training_abs_criteria: Optional[float] = None,
    model_training_test_ratio: float = 0.2,
    model_training_weight_decay: bool = False,
    model_training_stop_gradient: bool = False,
    model_loss_horizon: int = 10,
    model_horizon_fn: Callable[[int], int] = lambda epoch: 1,
    model_check_done_condition: bool = True,
    max_env_buffer_size: Optional[int] = None,
    max_model_buffer_size: Optional[int] = None,
    sac_learning_rate: float = 1e-4,
    sac_discounting: float = 0.99,
    sac_batch_size: int = 256,
    real_ratio: float = 0.06,
    sac_reward_scaling: float = 1.0,
    sac_tau: float = 0.005,
    sac_fixed_alpha: Optional[float] = None,
    sac_training_state: Optional[sac_base.TrainingState] = None,
    seed: int = 0,
    deterministic_in_env: bool = False,
    deterministic_eval: bool = False,
    model_network_factory: Callable = ssrl_networks.make_model_network,
    sac_network_factory: types.NetworkFactory[
        sac_networks.SACNetworks] = sac_networks.make_sac_networks,
    hallucination_max_std: Optional[float] = -1.0,  # if <= 0, use the same from the provided sac_network_factory # noqa: E501
    zero_final_layer_of_policy: bool = False,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None
):
    """ssrl training.

    The algorithm is as follows:
    for num_epochs:
        for model_trains_per_epoch:
            train the model
            for training_steps_per_model_train:
                for env_steps_per_training_step:
                    step the environment
                for hallucination_updates_per_training_step:
                    sample model_rollouts_per_hallucination_update states
                    do model_horizon rollouts from sampled states
                    do sac_grad_updates_per_hallucination_update sac updates

    Args:
    TODO
    """

    # make sure either num_timesteps or num_epochs is given
    assert num_timesteps is not None or num_epochs is not None

    ms = initizalize_training(
        environment=environment,
        low_level_control_fn=low_level_control_fn,
        dynamics_fn=dynamics_fn,
        reward_fn=reward_fn,
        model_output_dim=model_output_dim,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        policy_repeat=policy_repeat,
        num_epochs=num_epochs,
        model_trains_per_epoch=model_trains_per_epoch,
        training_steps_per_model_train=training_steps_per_model_train,
        env_steps_per_training_step=env_steps_per_training_step,
        hallucination_updates_per_training_step=hallucination_updates_per_training_step,  # noqa: E501
        model_rollouts_per_hallucination_update=model_rollouts_per_hallucination_update,  # noqa: E501
        sac_grad_updates_per_hallucination_update=sac_grad_updates_per_hallucination_update,  # noqa: E501
        init_exploration_steps=init_exploration_steps,
        clear_model_buffer_after_model_train=clear_model_buffer_after_model_train,  # noqa: E501
        action_repeat=action_repeat,
        obs_history_length=obs_history_length,
        num_envs=num_envs,
        num_evals=num_evals,
        num_eval_envs=num_eval_envs,
        policy_normalize_observations=policy_normalize_observations,
        model_learning_rate=model_learning_rate,
        model_training_batch_size=model_training_batch_size,
        model_training_max_sgd_steps_per_epoch=model_training_max_sgd_steps_per_epoch,  # noqa: E501
        model_training_max_epochs=model_training_max_epochs,
        model_training_convergence_criteria=model_training_convergence_criteria,  # noqa: E501
        model_training_consec_converged_epochs=model_training_consec_converged_epochs,  # noqa: E501
        model_training_abs_criteria=model_training_abs_criteria,
        model_training_test_ratio=model_training_test_ratio,
        model_training_weight_decay=model_training_weight_decay,
        model_training_stop_gradient=model_training_stop_gradient,
        model_loss_horizon=model_loss_horizon,
        model_horizon_fn=model_horizon_fn,
        model_check_done_condition=model_check_done_condition,
        max_env_buffer_size=max_env_buffer_size,
        max_model_buffer_size=max_model_buffer_size,
        sac_learning_rate=sac_learning_rate,
        sac_discounting=sac_discounting,
        sac_batch_size=sac_batch_size,
        real_ratio=real_ratio,
        sac_reward_scaling=sac_reward_scaling,
        sac_tau=sac_tau,
        sac_fixed_alpha=sac_fixed_alpha,
        sac_training_state=sac_training_state,
        seed=seed,
        deterministic_in_env=deterministic_in_env,
        deterministic_eval=deterministic_eval,
        model_network_factory=model_network_factory,
        sac_network_factory=sac_network_factory,
        hallucination_max_std=hallucination_max_std,
        zero_final_layer_of_policy=zero_final_layer_of_policy,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        eval_env=eval_env
    )

    # Run initial eval
    metrics = {}
    if num_evals > 1:
        metrics = ms.evaluator.run_evaluation(
            (ms.sac_state.training_state.normalizer_params,
             ms.sac_state.training_state.policy_params),
            training_metrics={})
        logging.info(metrics)
        progress_fn(0, metrics)
        policy_params_fn(
            0,
            ms.sac_state.constants.make_policy,
            (ms.sac_state.training_state.normalizer_params,
             ms.sac_state.training_state.policy_params),
            metrics)

    # Prefill the env buffer.
    def prefill_env_buffer(
        training_state: base.TrainingState,
        sac_ts: sac_base.TrainingState,
        env_state: envs.State,
        env_buffer_state: base.ReplayBufferState,
        model_buffer_state: base.ReplayBufferState,
        key: PRNGKey,
        deterministic: bool
    ) -> Tuple[base.TrainingState, sac_base.TrainingState, envs.State,
               base.ReplayBufferState, PRNGKey]:

        def f(carry: Tuple[base.TrainingState, sac_base.TrainingState,
                           envs.State, base.ReplayBufferState, PRNGKey],
              unused):
            del unused
            (training_state, sac_ts, env_state, model_buffer_state,
             key) = carry
            key, new_key = jax.random.split(key)
            (new_normalizer_params, env_state, model_buffer_state,
             transition) = get_experience(
                sac_ts.normalizer_params,
                sac_ts.policy_params,
                ms.constants.make_policy_env,
                env_state, model_buffer_state, key, ms.env,
                ms.sac_state.replay_buffer,
                deterministic)
            new_training_state = training_state.replace(
                env_steps=(training_state.env_steps
                           + ms.constants.action_repeat*ms.constants.num_envs))
            new_sac_ts = sac_ts.replace(
                normalizer_params=new_normalizer_params)
            return (new_training_state, new_sac_ts, env_state,
                    model_buffer_state, new_key), transition

        (training_state, sac_ts, env_state, model_buffer_state,
         key), transitions = jax.lax.scan(
            f,
            (training_state, sac_ts, env_state, model_buffer_state, key),
            (), length=ms.constants.init_exploration_steps)

        # we insert the transitions into the env buffer after the scan finishes
        # to ensure that they are inserted in order
        env_buffer_state = ms.env_buffer.insert(env_buffer_state, transitions)

        return (training_state, sac_ts, env_state, env_buffer_state,
                model_buffer_state, key)

    local_key, env_key, prefill_key = jax.random.split(ms.local_key, 3)
    env_keys = jax.random.split(env_key, num_envs)
    env_state = ms.env.reset(env_keys)
    ms = ms.replace(local_key=local_key)
    (training_state, sac_ts, env_state, env_buffer_state,
     model_buffer_state, _) = prefill_env_buffer(
        ms.training_state, ms.sac_state.training_state, env_state,
        ms.env_buffer_state, ms.sac_state.buffer_state, prefill_key,
        ms.constants.deterministic_in_env)
    ms = ms.replace(training_state=training_state,
                    sac_state=ms.sac_state.replace(
                        training_state=sac_ts,
                        buffer_state=model_buffer_state),
                    env_buffer_state=env_buffer_state)
    env_buffer_size = ms.env_buffer.size(env_buffer_state)
    logging.info('env buffer size after init exploration %s', env_buffer_size)

    # Training loop
    num_evals_after_init = max(num_evals - 1, 1)
    eval_threshold = ms.constants.num_epochs / num_evals_after_init
    training_walltime = 0
    for epoch in range(ms.constants.num_epochs):

        # Update model horizon
        ms = update_model_horizon(ms, epoch)

        # Run training epoch
        (ms, training_metrics, training_walltime,
         env_state) = sim_training_epoch_with_timing(
            ms, training_walltime, env_state)

        # Run evals evenly spaced between epochs
        if ((epoch != 0 or num_evals >= ms.constants.num_epochs)
            and (((epoch + 1) % eval_threshold <= 1
                  and (epoch + 1) / eval_threshold >= 1)
                 or epoch == ms.constants.num_epochs - 1)):
            current_step = ms.training_state.env_steps
            metrics = ms.evaluator.run_evaluation(
                (ms.sac_state.training_state.normalizer_params,
                 ms.sac_state.training_state.policy_params),
                training_metrics)
            logging.info(metrics)
            progress_fn(current_step, metrics)
            policy_params_fn(
                current_step,
                ms.sac_state.constants.make_policy,
                (ms.sac_state.training_state.normalizer_params,
                 ms.sac_state.training_state.policy_params),
                metrics)

    return (ms.training_state, ms.sac_state.training_state,
            ms.env_buffer_state)

    # return (ms.sac_state.constants.make_policy,
    #         (ms.sac_state.training_state.normalizer_params,
    #          ms.sac_state.training_state.policy_params),
    #         metrics)


def initizalize_training(
    environment: envs.Env,
    low_level_control_fn: base.low_level_control_fn,
    dynamics_fn: base.dynamics_fn,
    reward_fn: base.reward_fn,
    model_output_dim: int,
    episode_length: int,
    policy_repeat: int,
    num_timesteps,
    num_epochs: Optional[int],
    model_trains_per_epoch: int,
    training_steps_per_model_train: int,
    env_steps_per_training_step: int,
    hallucination_updates_per_training_step: int,
    model_rollouts_per_hallucination_update: int,
    sac_grad_updates_per_hallucination_update: int,
    init_exploration_steps: int,
    clear_model_buffer_after_model_train: bool,
    action_repeat: int,
    obs_history_length: int,
    num_envs: int,
    num_evals: int,
    num_eval_envs: int,
    policy_normalize_observations: bool,
    model_learning_rate: float,
    model_training_batch_size: int,
    model_training_max_sgd_steps_per_epoch: Optional[int],
    model_training_max_epochs: int,
    model_training_convergence_criteria: float,
    model_training_consec_converged_epochs: int,
    model_training_abs_criteria: Optional[float],
    model_training_test_ratio: float,
    model_training_weight_decay: bool,
    model_training_stop_gradient: bool,
    model_loss_horizon: int,
    model_horizon_fn: Callable[[int], int],
    model_check_done_condition: bool,
    max_env_buffer_size: Optional[int],
    max_model_buffer_size: Optional[int],
    sac_learning_rate: float,
    sac_discounting: float,
    sac_batch_size: int,
    real_ratio: float,
    sac_reward_scaling: float,
    sac_tau: float,
    sac_fixed_alpha: Optional[float],
    sac_training_state: Optional[sac_base.TrainingState],
    seed: int,
    deterministic_in_env: bool,
    deterministic_eval: bool,
    model_network_factory: Callable,
    sac_network_factory: types.NetworkFactory[
        sac_networks.SACNetworks],
    hallucination_max_std: Optional[float],
    zero_final_layer_of_policy: bool,
    progress_fn: Callable[[int, Metrics], None],
    policy_params_fn: Callable[..., None],
    eval_env: Optional[envs.Env]
) -> base.MbpoState:

    # assert (real_ratio * sac_grad_updates_per_step * sac_batch_size
    #         <= model_rollouts_per_env_step)

    env = envs.training.wrap(
        environment, episode_length=episode_length,
        action_repeat=action_repeat,
        obs_history_length=obs_history_length)

    env_steps_per_epoch = (model_trains_per_epoch
                           * training_steps_per_model_train
                           * env_steps_per_training_step
                           * num_envs
                           * action_repeat)

    if num_epochs is None:
        num_epochs = math.ceil(
            (num_timesteps - init_exploration_steps)
            // env_steps_per_epoch + 1
        )

    if max_env_buffer_size is None:
        max_env_buffer_size = (
            env_steps_per_epoch * num_epochs + init_exploration_steps)

    total_batches = max_env_buffer_size // model_training_batch_size
    model_max_train_batches = int(
        jp.ceil(total_batches * (1 - model_training_test_ratio)))
    model_max_test_batches = total_batches - model_max_train_batches + 1

    if isinstance(hallucination_updates_per_training_step, int):
        hallucination_updates_per_training_step_fn = (
            lambda epoch: hallucination_updates_per_training_step)
    elif callable(hallucination_updates_per_training_step):
        hallucination_updates_per_training_step_fn = (
            hallucination_updates_per_training_step)
    else:
        raise ValueError('hallucination_updates_per_training_step must be an '
                         'int or a callable')

    constants = base.Constants(
        num_epochs=num_epochs,
        model_trains_per_epoch=model_trains_per_epoch,
        training_steps_per_model_train=training_steps_per_model_train,
        env_steps_per_training_step=env_steps_per_training_step,
        hallucination_updates_per_training_step_fn=hallucination_updates_per_training_step_fn,  # noqa: E501
        model_rollouts_per_hallucination_update=model_rollouts_per_hallucination_update,  # noqa: E501
        sac_grad_updates_per_hallucination_update=sac_grad_updates_per_hallucination_update,  # noqa: E501
        init_exploration_steps=init_exploration_steps,
        env_steps_per_epoch=env_steps_per_epoch,
        clear_model_buffer_after_model_train=clear_model_buffer_after_model_train,  # noqa: E501
        num_envs=num_envs,
        obs_size=env.observation_size,
        obs_hist_len=obs_history_length,
        action_size=env.action_size,
        action_repeat=action_repeat,
        model_learning_rate=model_learning_rate,
        model_training_batch_size=model_training_batch_size,
        model_training_max_sgd_steps_per_epoch=model_training_max_sgd_steps_per_epoch,  # noqa: E501
        model_training_max_epochs=model_training_max_epochs,
        model_training_convergence_criteria=model_training_convergence_criteria,  # noqa: E501
        model_training_consec_converged_epochs=model_training_consec_converged_epochs,  # noqa: E501
        model_training_abs_criteria=model_training_abs_criteria,
        model_training_test_ratio=model_training_test_ratio,
        model_training_weight_decay=model_training_weight_decay,
        model_training_stop_gradient=model_training_stop_gradient,
        model_loss_horizon=model_loss_horizon,
        model_horizon_fn=model_horizon_fn,
        model_ensemble_size=0,
        model_num_elites=0,
        model_probabilistic=False,
        model_max_train_batches=model_max_train_batches,
        model_max_test_batches=model_max_test_batches,
        make_model=None,
        model_loss=None,
        sac_batch_size=sac_batch_size,
        real_ratio=real_ratio,
        model_update=None,
        low_level_control_fn=low_level_control_fn,
        dynamics_fn=dynamics_fn,
        reward_fn=reward_fn,
        policy_repeat=policy_repeat,
        max_model_buffer_size=max_model_buffer_size,
        deterministic_in_env=deterministic_in_env,
        make_policy_env=None,
    )

    global_key, local_key = jax.random.split(jax.random.PRNGKey(seed))

    # initialize model
    scale_fn = base.Scaler.transform
    model_network = model_network_factory(constants.obs_size, model_output_dim)
    constants = constants.replace(
        model_ensemble_size=model_network.ensemble_size,
        model_num_elites=model_network.num_elites,
        model_probabilistic=model_network.probabilistic)
    model_optimizer = optax.adam(model_learning_rate)
    make_model = ssrl_networks.make_inference_fn(model_network, scale_fn,
                                                   constants)
    training_state = _init_training_state(
        global_key, model_network, model_optimizer, constants)
    del global_key
    model_loss = ssrl_losses.make_losses(model_network, scale_fn, constants)
    model_update = gradients.gradient_update_fn(model_loss, model_optimizer,
                                                pmap_axis_name=None,
                                                has_aux=True)
    constants = constants.replace(
        model_update=model_update,
        make_model=make_model,
        model_loss=model_loss)

    # create the model_env
    done_fn = lambda *args: jp.zeros(())  # noqa: E731
    if model_check_done_condition:
        assert hasattr(env, 'is_done'), (
            'The environment must have an is_done method to check if the '
            'episode is done. Otherwise, set model_check_done_condition=False')
        done_fn = env.is_done
    model_env = base.ModelEnv(done_fn, constants.obs_size,
                              constants.action_size)
    model_env = envs.training.wrap(
        model_env, episode_length=episode_length,
        action_repeat=action_repeat,
        obs_history_length=obs_history_length,
        reset_at_episode_end=False)

    # make the env policy
    normalize_fn = lambda x, y: x  # noqa: E731
    if policy_normalize_observations:
        normalize_fn = running_statistics.normalize
    env_network = sac_network_factory(
        observation_size=env.observation_size*obs_history_length,
        action_size=env.action_size,
        preprocess_observations_fn=normalize_fn)
    make_policy_env = sac_networks.make_inference_fn(env_network)
    constants = constants.replace(make_policy_env=make_policy_env)

    # update sac_network_factory to with the hallucination_max_std
    if hallucination_max_std is None or hallucination_max_std > 0:
        sac_network_factory = functools.partial(
            sac_network_factory, policy_max_std=hallucination_max_std)

    # initialize buffers
    dummy_obs = jp.zeros((constants.obs_size * constants.obs_hist_len,))
    dummy_action = jp.zeros((constants.action_size,))
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
    env_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_env_buffer_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=model_rollouts_per_hallucination_update)
    local_key, eb_key = jax.random.split(local_key)
    env_buffer_state = env_buffer.init(eb_key)

    # initialize SAC
    init_model_horizon = constants.model_horizon_fn(0)
    init_hallucination_updates_per_training_step = (
        constants.hallucination_updates_per_training_step_fn(0))
    if init_model_horizon > 0:
        sac_max_replay_size = (
            training_steps_per_model_train
            * init_hallucination_updates_per_training_step
            * model_rollouts_per_hallucination_update
            * init_model_horizon)
    else:
        sac_max_replay_size = max_env_buffer_size
    if max_model_buffer_size is not None:
        sac_max_replay_size = min(sac_max_replay_size, max_model_buffer_size)

    sac_state = sac.initizalize_training(
        environment=environment,
        num_timesteps=episode_length,
        episode_length=episode_length,
        action_repeat=action_repeat,
        obs_history_length=obs_history_length,
        num_envs=1,
        num_eval_envs=1,
        learning_rate=sac_learning_rate,
        discounting=sac_discounting,
        seed=seed,
        batch_size=sac_batch_size,
        num_evals=1,
        normalize_observations=policy_normalize_observations,
        reward_scaling=sac_reward_scaling,
        tau=sac_tau,
        fixed_alpha=sac_fixed_alpha,
        min_replay_size=0,
        max_replay_size=sac_max_replay_size,
        grad_updates_per_step=sac_grad_updates_per_hallucination_update,
        deterministic_eval=deterministic_eval,
        network_factory=sac_network_factory,
        progress_fn=None,
        eval_env=None
    )

    if zero_final_layer_of_policy:
        policy_params = sac_state.training_state.policy_params
        last_key = list(policy_params['params'].keys())[-1]
        zeros = jp.zeros_like(policy_params['params'][last_key]['kernel'])
        policy_params['params'][last_key]['kernel'] = zeros
        bias = policy_params['params'][last_key]['bias']
        means, stds = jp.split(bias, 2)
        stds = -2.5 * jp.ones_like(stds)
        bias = jp.concatenate([means, stds])
        policy_params['params'][last_key]['bias'] = bias
        sac_state = sac_state.replace(
            training_state=sac_state.training_state.replace(
                policy_params=policy_params))

    # fine-tune from sac_training_state, if provided
    if sac_training_state is not None:
        sac_state = sac_state.replace(training_state=sac_training_state)

    if not eval_env:
        eval_env = env
    else:
        eval_env = envs.training.wrap(
            eval_env, episode_length=episode_length,
            action_repeat=action_repeat,
            obs_history_length=obs_history_length)

    local_key, eval_key = jax.random.split(local_key)
    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(constants.make_policy_env,
                          deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key)

    ssrl_state = base.MbpoState(
        training_state=training_state,
        sac_state=sac_state,
        constants=constants,
        env=env,
        env_buffer=env_buffer,
        env_buffer_state=env_buffer_state,
        model_env=model_env,
        evaluator=evaluator,
        local_key=local_key,
        model_horizon=constants.model_horizon_fn(0),
        hallucination_updates_per_training_step=(
            constants.hallucination_updates_per_training_step_fn(0))
    )

    return ssrl_state


# this method is not jittable due to the variable size of the env buffer coming
# in and because the number of training epochs is variable. However, each
# training epoch is jitted.
def train_model(
    training_state: base.TrainingState,
    env_buffer_state: base.ReplayBufferState,
    env_buffer: replay_buffers.UniformSamplingQueue,
    c: base.Constants,
    key: PRNGKey
):
    # Create a dataset using all transitions from the env buffer
    buffer_size = env_buffer.size(env_buffer_state)
    data = env_buffer_state.data[:buffer_size]
    all_transitions = env_buffer._unflatten_fn(data)
    dataset = base.Dataset(all_transitions, c.model_loss_horizon)

    # create shuffled idxs and split them into training and test sets where the
    # length of the train set is divisible by the batch size and the length of
    # the test set is as close as possible to the test ratio
    key_shuffle, key = jax.random.split(key)
    per_idxs = jax.random.permutation(key_shuffle, jp.arange(buffer_size))
    (train_length, test_length, num_train_batches,
     num_test_batches) = _calculate_dataset_lengths(
        buffer_size, c.model_training_batch_size, c.model_training_test_ratio)
    if c.model_training_max_sgd_steps_per_epoch is not None:
        num_train_batches = min(num_train_batches,
                                c.model_training_max_sgd_steps_per_epoch)
    train_idxs = per_idxs[:train_length]
    test_idxs = per_idxs[train_length:(train_length + test_length)]
    transitions = dataset[train_idxs]
    test_transitions = dataset[test_idxs]

    # scale the data
    all_train_data = data[train_idxs]
    all_train_transitions = env_buffer._unflatten_fn(all_train_data)
    train_obs = all_train_transitions.observation
    train_act = all_train_transitions.action
    scaler_params = base.Scaler.fit(train_obs, train_act)
    training_state = training_state.replace(scaler_params=scaler_params)

    params = training_state.model_params
    opt_state = training_state.model_optimizer_state
    best = 10e9 * jp.ones(c.model_ensemble_size)
    epochs_since_update = 0
    t = time.time()
    for epoch in range(c.model_training_max_epochs):
        key, epoch_key = jax.random.split(key)
        (params, opt_state, train_total_loss, train_mean_loss,
            test_total_loss, test_mean_loss) = model_training_epoch(
            params, opt_state, scaler_params, c, transitions,
            test_transitions, num_train_batches, num_test_batches, epoch_key)
        print(f'Model epoch {epoch}: train total loss {train_total_loss}, '
              f'train mean loss {train_mean_loss}, '
              f'test mean loss {test_mean_loss}')

        # check absolute criteria
        if (c.model_training_abs_criteria is not None
                and jp.sum(test_mean_loss < c.model_training_abs_criteria)
                >= c.model_num_elites):
            break

        # check convergence criteria
        improvement = (best - test_mean_loss) / best
        best = jp.where(improvement > c.model_training_convergence_criteria,
                        test_mean_loss, best)
        if jp.any(improvement > c.model_training_convergence_criteria):
            epochs_since_update = 0
        else:
            epochs_since_update += 1
        if epochs_since_update >= c.model_training_consec_converged_epochs:
            break

    sec_per_epoch = (time.time() - t) / (epoch + 1)

    elite_idxs = jp.argsort(test_mean_loss)
    elite_idxs = elite_idxs[:c.model_num_elites]
    params['elites']['idxs'] = elite_idxs

    training_state = training_state.replace(
        model_params=params, model_optimizer_state=opt_state)

    metrics = {'train_total_loss': train_total_loss,
               'train_mean_loss': train_mean_loss,
               'test_total_loss': test_total_loss,
               'test_mean_loss': jp.mean(test_mean_loss),
               'train_epochs': epoch + 1,
               'sec_per_epoch': sec_per_epoch}

    print(f'Model trained in {epoch + 1} epochs '
          f'with {buffer_size} transitions.')
    if epoch + 1 == c.model_training_max_epochs:
        print('Warning: model training did not converge within'
              f'{c.model_training_max_epochs} epochs.')

    return training_state, metrics


def _calculate_dataset_lengths(D: int, B: int, test_set_ratio: float):
    """Calculate the lengths of the training and test sets to be as close as
    possible to the test_set_ratio, ensuring that the training set length is
    divisible by B.

    Args:
    D: int, total number of samples in the dataset
    B: int, batch size
    test_set_ratio: float, ratio of the test set length to the total dataset
    """

    total_batches = D // B
    test_batches = int(jp.floor(total_batches * test_set_ratio))
    train_batches = total_batches - test_batches
    train_length = train_batches * B
    test_length = test_batches * B

    return train_length, test_length, train_batches, test_batches


def model_training_epoch(
    params: Params,
    model_optimizer_state: optax.OptState,
    scaler_params: base.ScalerParams,
    c: base.Constants,
    transitions: types.Transition,
    test_transitions: types.Transition,
    num_train_batches: int,
    num_test_batches: int,
    key: PRNGKey
):
    transitions, test_transitions = prepare_data(
        c, transitions, test_transitions, key)

    (new_params, new_opt_state, train_total_losses, train_mean_losses,
     test_total_losses, test_mean_losses) = model_training_epoch_jit(
        params, model_optimizer_state, scaler_params, c, transitions,
        test_transitions, num_train_batches, num_test_batches)

    return (new_params, new_opt_state,
            train_total_losses[num_train_batches-1],
            train_mean_losses[num_train_batches-1],
            jp.mean(test_total_losses[:num_test_batches]),
            jp.mean(test_mean_losses[:num_test_batches], axis=0))


def prepare_data(
    c: base.Constants,
    transitions: types.Transition,
    test_transitions: types.Transition,
    key: PRNGKey
):
    # (num_samples, horizon, dim) -> (ensemble_size, num_samples, horizon, dim)
    # by duplicating the data for each model in the ensemble
    def duplicate_for_ensemble(transitions: types.Transition):
        num_samples = transitions.observation.shape[0]
        transitions = jax.tree_map(lambda x: jp.expand_dims(x, axis=0),
                                   transitions)
        transitions = jax.tree_map(
            lambda x: jp.broadcast_to(
                x, (c.model_ensemble_size, num_samples) + x.shape[2:]),
            transitions)
        return transitions
    transitions = duplicate_for_ensemble(transitions)
    test_transitions = duplicate_for_ensemble(test_transitions)

    # shuffle the data for each model in the ensemble (the sequence of data
    # that each model sees will be different)
    def shuffle_subarr(subarr, key):
        permuted_idxs = jax.random.permutation(key, subarr.shape[0])
        return subarr[permuted_idxs]
    keys = jax.random.split(key, c.model_ensemble_size)
    transitions = jax.tree_map(lambda x: jax.vmap(shuffle_subarr)(x, keys),
                               transitions)

    # put data into batches: reshape to
    # (num_batches, batch_size, ensemble_size, horizon, dim)
    transitions = jax.tree_map(
        lambda x: x.reshape(
            ((-1, c.model_training_batch_size, c.model_ensemble_size)
             + x.shape[2:])),
        transitions)
    test_transitions = jax.tree_map(
        lambda x: x.reshape(
            ((-1, c.model_training_batch_size, c.model_ensemble_size)
             + x.shape[2:])),
        test_transitions)

    # transpose to (num_batches, batch_size, horizon, ensemble_size, dim)
    transitions = jax.tree_map(lambda x: jp.swapaxes(x, 2, 3), transitions)
    test_transitions = jax.tree_map(lambda x: jp.swapaxes(x, 2, 3),
                                    test_transitions)

    # expand to (max_batches, batch_size, horizon, ensemble_size, dim)
    def expand(arr, leading_dim):
        expanded_array_shape = (leading_dim,) + arr.shape[1:]
        expanded_array = jp.zeros(expanded_array_shape)
        expanded_array = expanded_array.at[:arr.shape[0]].set(arr)
        return expanded_array
    transitions = jax.tree_map(
        lambda x: expand(x, c.model_max_train_batches), transitions)
    test_transitions = jax.tree_map(
        lambda x: expand(x, c.model_max_test_batches), test_transitions)

    return transitions, test_transitions


@functools.partial(jax.jit, static_argnames=['c'])
def model_training_epoch_jit(
    params: Params,
    model_optimizer_state: optax.OptState,
    scaler_params: base.ScalerParams,
    c: base.Constants,
    transitions: types.Transition,
    test_transitions: types.Transition,
    num_train_batches: int,
    num_test_batches: int
):
    def sgd_step(carry, in_element):
        params, opt_state = carry
        transitions, i = in_element
        model_params = params.pop('params')
        (loss, loss_ensemble), new_model_params, opt_state = jax.lax.cond(
            i < num_train_batches,
            lambda: c.model_update(
                model_params,
                params,
                scaler_params,
                transitions.observation,
                transitions.next_observation,
                transitions.action,
                transitions.discount,
                optimizer_state=opt_state
            ),
            lambda: ((0., jp.zeros(c.model_ensemble_size)),
                     model_params, opt_state),
        )
        return (({'params': new_model_params, **params}, opt_state),
                (loss, loss_ensemble))

    # perform an sgd step for each batch
    ((new_params, new_opt_state),
     (train_total_losses, mean_losses)) = jax.lax.scan(
        sgd_step, (params, model_optimizer_state),
        (transitions, jp.arange(transitions.observation.shape[0])))
    train_mean_losses = jp.mean(mean_losses, axis=-1)

    # compute lossses on the test set
    new_model_params = new_params.pop('params')

    def test(_, in_element):
        transitions, i = in_element
        test_total_loss, test_mean_loss = jax.lax.cond(
            i < num_test_batches,
            lambda: c.model_loss(
                new_model_params, new_params, scaler_params,
                transitions.observation, transitions.next_observation,
                transitions.action, transitions.discount),
            lambda: (0., jp.zeros(c.model_ensemble_size)),
        )
        return None, (test_total_loss, test_mean_loss)

    _, (test_total_losses, test_mean_losses) = jax.lax.scan(
        test, None,
        (test_transitions, jp.arange(test_transitions.observation.shape[0])))
    new_params = {'params': new_model_params, **new_params}

    return (new_params, new_opt_state, train_total_losses, train_mean_losses,
            test_total_losses, test_mean_losses)


def _init_training_state(
    key: PRNGKey,
    model_network: ssrl_networks.EnsembleModel,
    model_optimizer: optax.GradientTransformation,
    c: base.Constants
) -> base.TrainingState:
    dummy_X = jp.zeros((model_network.ensemble_size,
                        c.obs_size*c.obs_hist_len + c.action_size))
    model_params = model_network.init(key, dummy_X)
    model_optimizer_state = model_optimizer.init(model_params['params'])

    scaler_params = base.Scaler.init(c.obs_size*c.obs_hist_len, c.action_size)

    training_state = base.TrainingState(
        model_optimizer_state=model_optimizer_state,
        model_params=model_params,
        scaler_params=scaler_params,
        env_steps=jp.zeros(())
    )

    return training_state


def get_experience(
    normalizer_params: running_statistics.RunningStatisticsState,
    policy_params: Params,
    make_policy: Callable,
    env_state: envs.State,
    model_buffer_state: base.ReplayBufferState,
    key: PRNGKey,
    env: envs.Env,
    model_replay_buffer: replay_buffers.UniformSamplingQueue,
    deterministic: bool
) -> Tuple[running_statistics.RunningStatisticsState,
           envs.State, base.ReplayBufferState]:

    policy = make_policy((normalizer_params, policy_params),
                         deterministic=deterministic)
    env_state, transitions = acting.actor_step(
        env, env_state, policy, key, extra_fields=('truncation',))

    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation)

    model_buffer_state = model_replay_buffer.insert(model_buffer_state,
                                                    transitions)
    return normalizer_params, env_state, model_buffer_state, transitions


# Note that this is NOT a pure jittable method.
def sim_training_epoch_with_timing(
    ms: base.MbpoState,
    training_walltime: float,
    env_state: envs.State
):
    t = time.time()
    model_train_time = 0
    other_time = 0
    model_metrics = {}
    for _ in range(ms.constants.model_trains_per_epoch):
        # train model
        if ms.constants.model_training_max_epochs > 0:
            new_key, model_key = jax.random.split(ms.local_key)
            ms = ms.replace(local_key=new_key)
            start_time = time.time()
            training_state, model_metrics = train_model(
                ms.training_state, ms.env_buffer_state, ms.env_buffer,
                ms.constants, model_key)
            ms = ms.replace(training_state=training_state)
            model_train_time += time.time() - start_time

            if ms.constants.clear_model_buffer_after_model_train:
                rb = ms.sac_state.replay_buffer
                bs = rb.init(ms.sac_state.buffer_state.key)
                ms = ms.replace(sac_state=ms.sac_state.replace(buffer_state=bs))

        # do env steps, hallucinations, and sac training
        start_time = time.time()
        epoch_key, new_local_key = jax.random.split(ms.local_key)
        ms = ms.replace(local_key=new_local_key)
        (training_state, sac_training_state, env_state, env_buffer_state,
         sac_buffer_state, sac_metrics) = sim_training_epoch(
            ms.training_state,
            ms.sac_state.training_state,
            env_state,
            ms.env_buffer_state, epoch_key,
            ms.constants, ms.sac_state.constants,
            ms.env, ms.env_buffer,
            ms.sac_state.buffer_state,
            ms.sac_state.replay_buffer,
            ms.model_env,
            ms.model_horizon,
            ms.hallucination_updates_per_training_step
        )
        other_time += time.time() - start_time
        ms = ms.replace(
            training_state=training_state,
            sac_state=ms.sac_state.replace(training_state=sac_training_state,
                                           buffer_state=sac_buffer_state),
            env_buffer_state=env_buffer_state)

    sac_metrics = jax.tree_util.tree_map(jp.mean, sac_metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), sac_metrics)
    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = ms.constants.env_steps_per_epoch / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        'training/model_train_time': model_train_time,
        'training/other_time': other_time,
        'training/model_horizon': ms.model_horizon,
        'training/hallucination_updates_per_training_step': (
            ms.hallucination_updates_per_training_step),
        'training/env_buffer_size': ms.env_buffer.size(ms.env_buffer_state),
        **{f'model/{name}': value for name, value in model_metrics.items()},
        **{f'sac/{name}': value for name, value in sac_metrics.items()}
    }
    return ms, metrics, training_walltime, env_state


@functools.partial(jax.jit, static_argnames=[
    'c', 'sc', 'env', 'env_buffer', 'sac_buffer', 'model_env', 'model_horizon',
    'hallucination_updates_per_training_step'
])
def sim_training_epoch(
    training_state: base.TrainingState,
    sac_training_state: sac_base.TrainingState,
    env_state: envs.State,
    env_buffer_state: base.ReplayBufferState,
    key: PRNGKey,
    c: base.Constants,
    sc: sac_base.Constants,
    env: envs.Env,
    env_buffer: replay_buffers.UniformSamplingQueue,
    sac_buffer_state: base.ReplayBufferState,
    sac_buffer: replay_buffers.UniformSamplingQueue,
    model_env: base.ModelEnv,
    model_horizon: int,
    hallucination_updates_per_training_step: int
) -> Tuple[base.TrainingState, sac_base.TrainingState,
           envs.State, base.ReplayBufferState, Metrics]:

    # training step
    def f(carry, unused_t):
        ts, sts, es, ebs, sbs, k = carry
        k, new_key = jax.random.split(k)
        ts, sts, es, ebs, sbs, metrics = sim_training_step(
            ts, sts, es, ebs, k, c, sc, env, env_buffer, sbs, sac_buffer,
            model_env, model_horizon, hallucination_updates_per_training_step)
        return (ts, sts, es, ebs, sbs, new_key), metrics

    (ts, sts, es, ebs, sbs, key), metrics = jax.lax.scan(
        f,
        (training_state, sac_training_state, env_state, env_buffer_state,
         sac_buffer_state, key),
        (), length=c.training_steps_per_model_train)
    metrics = jax.tree_util.tree_map(jp.mean, metrics)

    return ts, sts, es, ebs, sbs, metrics


def sim_training_step(
    training_state: base.TrainingState,
    sac_training_state: sac_base.TrainingState,
    env_state: envs.State,
    env_buffer_state: base.ReplayBufferState,
    key: PRNGKey,
    c: base.Constants,
    sc: sac_base.Constants,
    env: envs.Env,
    env_buffer: replay_buffers.UniformSamplingQueue,
    sac_buffer_state: base.ReplayBufferState,
    sac_buffer: replay_buffers.UniformSamplingQueue,
    model_env: base.ModelEnv,
    model_horizon: int,
    hallucination_updates_per_training_step: int
) -> Tuple[base.TrainingState, sac_base.TrainingState, envs.State,
           base.ReplayBufferState, base.ReplayBufferState, Metrics]:

    # get experience
    def f(carry, unused_t):
        (ts, normalizer_params, env_state, sac_buffer_state,
         key) = carry
        key, new_key = jax.random.split(key)
        (normalizer_params, env_state, sac_buffer_state,
         transition) = get_experience(
            normalizer_params, sac_training_state.policy_params,
            c.make_policy_env, env_state, sac_buffer_state,
            key, env, sac_buffer, c.deterministic_in_env)
        new_ts = ts.replace(
            env_steps=(ts.env_steps + c.action_repeat*c.num_envs))
        return (new_ts, normalizer_params, env_state,
                sac_buffer_state, new_key), transition

    (training_state, normalizer_params, env_state,
     sac_buffer_state, key), transitions = jax.lax.scan(
        f,
        (training_state, sac_training_state.normalizer_params, env_state,
         sac_buffer_state, key), (),
         length=c.env_steps_per_training_step)

    # we insert the transitions into the env buffer after the scan finishes to
    # ensure that they are inserted in order
    env_buffer_state = env_buffer.insert(env_buffer_state, transitions)

    sac_training_state = sac_training_state.replace(
        normalizer_params=normalizer_params)

    # policy update
    (training_state, sac_training_state, env_buffer_state, sac_buffer_state,
     metrics) = policy_update(
        training_state, sac_training_state, c, sc,
        env_buffer_state, env_buffer, sac_buffer_state, sac_buffer, model_env,
        key, model_horizon, hallucination_updates_per_training_step)

    return (training_state, sac_training_state, env_state, env_buffer_state,
            sac_buffer_state, metrics)


@functools.partial(jax.jit, static_argnames=[
    'c', 'sc', 'env_buffer', 'sac_buffer', 'model_env', 'model_horizon',
    'hallucination_updates_per_training_step'])
def policy_update(
    training_state: base.TrainingState,
    sac_training_state: sac_base.TrainingState,
    c: base.Constants,
    sc: sac_base.Constants,
    env_buffer_state: base.ReplayBufferState,
    env_buffer: replay_buffers.UniformSamplingQueue,
    sac_buffer_state: base.ReplayBufferState,
    sac_buffer: replay_buffers.UniformSamplingQueue,
    model_env: base.ModelEnv,
    key: PRNGKey,
    model_horizon: int,
    hallucination_updates_per_training_step: int
) -> Tuple[base.TrainingState, sac_base.TrainingState, Metrics]:

    def update(carry, unused_t):
        sac_training_state, sac_buffer_state, env_buffer_state, key = carry

        # sample model_rollouts_per_hallucination_update samples from env
        # buffer
        env_buffer_state, transitions = env_buffer.sample(env_buffer_state)

        # we do the following to get the first state of each trajectory from
        # the sampled transitions (in place of calling an env.reset function;
        # trajectories that are done will reset to their sampled state)
        obs_stack = transitions.observation
        keys = jax.random.split(key, obs_stack.shape[0])

        def init_starting_state(obs_stack, key):
            return envs.State(
                pipeline_state=None,
                obs=obs_stack,
                reward=jp.zeros(()),
                done=jp.zeros(()),
                info={
                    'reward': jp.zeros(()),
                    'next_obs': jp.zeros((model_env.observation_size,)),
                    'first_pipeline_state': None,
                    'first_obs': obs_stack,
                    'first_metrics': {},
                    'truncation': jp.zeros(()),
                    'steps': jp.zeros(()),
                },
                prev_obs=obs_stack,
            )
        env_states = jax.vmap(init_starting_state)(obs_stack, keys)

        # perform model_horizon step model rollouts from samples and add to
        # model buffer (the model buffer is the sac buffer)
        def f(carry, unused_t):
            env_state, model_buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, model_buffer_state = get_experience_model(
                sac_training_state.normalizer_params,
                sac_training_state.policy_params,
                sc.make_policy,
                training_state.scaler_params,
                training_state.model_params,
                c.make_model,
                model_env, env_state, model_buffer_state,
                sac_buffer, key)
            return (env_state, model_buffer_state, new_key), {}

        _, sac_buffer_state, key = jax.lax.scan(
            f, (env_states, sac_buffer_state, key), (),
            length=model_horizon)[0]

        # update policy using sac
        (sac_training_state, sac_buffer_state, metrics,
         env_buffer_state) = sac.policy_update(
            sac_buffer, sac_buffer_state, sac_training_state, key, sc,
            env_buffer, env_buffer_state, c.real_ratio)

        return (sac_training_state, sac_buffer_state, env_buffer_state,
                key), metrics

    (sac_training_state, sac_buffer_state, env_buffer_state,
     _), metrics = jax.lax.scan(
        update,
        (sac_training_state, sac_buffer_state, env_buffer_state, key),
        (), length=hallucination_updates_per_training_step)

    metrics = jax.tree_util.tree_map(jp.mean, metrics)

    return (training_state, sac_training_state, env_buffer_state,
            sac_buffer_state, metrics)


def get_experience_model(
    normalizer_params: running_statistics.RunningStatisticsState,
    policy_params: Params,
    make_policy: Callable,
    model_scaler_params: base.ScalerParams,
    model_params: Params,
    make_model: Callable,
    model_env: base.ModelEnv,
    env_state: envs.State,
    model_buffer_state: base.ReplayBufferState,
    model_replay_buffer: replay_buffers.UniformSamplingQueue,
    key: PRNGKey
):
    policy = make_policy((normalizer_params, policy_params))
    model = make_model((model_scaler_params, model_params))
    env_state, transitions = model_actor_step(
        model_env, env_state, policy, model, key,
        extra_fields=('truncation',))
    model_buffer_state = model_replay_buffer.insert(model_buffer_state,
                                                    transitions)

    return env_state, model_buffer_state


def model_actor_step(
    env: envs.Env,
    env_state: acting.State,
    policy: Policy,
    model: base.Model,
    key: PRNGKey,
    extra_fields: Sequence[str] = ()
) -> Tuple[envs.State, Transition]:
    key_policy, key = jax.random.split(key)
    actions, policy_extras = policy(env_state.obs, key_policy)
    obs_stack = env_state.obs
    model_keys = jax.random.split(key, obs_stack.shape[0])
    next_obs, reward = jax.vmap(model)(obs_stack, actions, model_keys)
    info = env_state.info
    info['next_obs'] = next_obs
    info['reward'] = reward
    env_state = env_state.replace(info=info)
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, Transition(
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={
            'policy_extras': policy_extras,
            'state_extras': state_extras
        })


def update_model_horizon(ms: base.MbpoState, epoch: int) -> base.MbpoState:
    # here we update the model horizon using the model_horizon_fn. Also, the
    # hallucination_updates_per_training_step is updated. NOTE: if the
    # model_horizon_fn changes the model horizon from the previous epoch, some
    # jitted functions will need to be recompiled. Also, the SAC buffer is
    # resized to accomodate the new model horizon (while keeping old samples in
    # the buffer).
    model_horizon = ms.constants.model_horizon_fn(epoch)
    hallucination_updates_per_training_step = (
        ms.constants.hallucination_updates_per_training_step_fn(epoch))
    if ((model_horizon != ms.model_horizon
         or hallucination_updates_per_training_step != ms.hallucination_updates_per_training_step)  # noqa: E501
            and model_horizon != 0):
        key_rb, key = jax.random.split(ms.local_key)
        if model_horizon > 0:
            sac_max_replay_size = (
                ms.constants.training_steps_per_model_train
                * hallucination_updates_per_training_step
                * ms.constants.model_rollouts_per_hallucination_update
                * model_horizon)
        else:
            sac_max_replay_size = ms.constants.max_env_buffer_size
        if ms.constants.max_model_buffer_size is not None:
            sac_max_replay_size = min(sac_max_replay_size,
                                      ms.constants.max_model_buffer_size)
        dummy_obs = jp.zeros((ms.constants.obs_size
                              * ms.constants.obs_hist_len,))
        dummy_action = jp.zeros((ms.constants.action_size,))
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
        sac_replay_buffer = replay_buffers.UniformSamplingQueue(
            max_replay_size=sac_max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=(
                ms.constants.sac_batch_size
                * ms.constants.sac_grad_updates_per_hallucination_update))
        sac_buffer_state = sac_replay_buffer.init(key_rb)

        if not ms.constants.clear_model_buffer_after_model_train:
            current_size = ms.sac_state.replay_buffer.size(
                ms.sac_state.buffer_state)
            current_data = ms.sac_state.buffer_state.data[:current_size]
            sac_buffer_state = sac_buffer_state.replace(
                data=sac_buffer_state.data.at[:current_size].set(current_data),
                insert_position=ms.sac_state.buffer_state.insert_position,
                sample_position=ms.sac_state.buffer_state.sample_position
            )

        ms = ms.replace(
            model_horizon=model_horizon,
            hallucination_updates_per_training_step=(
                hallucination_updates_per_training_step),
            local_key=key,
            sac_state=ms.sac_state.replace(
                buffer_state=sac_buffer_state,
                replay_buffer=sac_replay_buffer)
        )
        print(f'Model horizon updated to {model_horizon}.')
        print(f'Hallucination updates per training step updated to '
              f'{hallucination_updates_per_training_step}.')
        print(f'SAC buffer resized to {sac_max_replay_size} samples.')
    return ms


def make_linear_threshold_fn(
    start_epoch: int,
    end_epoch: int,
    start_model_horizon: int,
    end_model_horizon: int
) -> Callable[[int], float]:
    a = start_epoch
    b = end_epoch
    x = start_model_horizon
    y = end_model_horizon

    def f(epoch):
        return math.floor(min(max(x + (epoch - a)/(b - a)*(y - x), x), y))

    return f

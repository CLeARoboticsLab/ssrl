from brax import envs
from brax.training import types
from brax.training import acting
from brax.envs.wrappers.training import FrameStackWrapper, EpisodeWrapper, AutoResetWrapper
import jax
import functools


def evaluate(params, env_unwrapped: envs.Env,
             make_policy,
             episode_length: int,
             action_repeat: int,
             key: types.PRNGKey,
             obs_history_length: int = 1,
             deterministic: bool = True,
             jit: bool = True):
    """Evaluates the policy by unrolling on the environment for episode_length
    steps and returning a list of pipeline states that can be used for
    rendering."""

    # create policy and env functions
    if len(params) == 2:
        normalizer_params, policy_params = params
    elif len(params) == 3:
        normalizer_params, policy_params, value_params = params
    policy = make_policy((normalizer_params, policy_params), deterministic=deterministic)
    env = FrameStackWrapper(env_unwrapped, obs_history_length=obs_history_length)
    env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
    env = AutoResetWrapper(env)
    env_reset = env.reset
    env_step = env.step

    if jit:
        policy = jax.jit(policy)
        env_reset = jax.jit(env.reset)
        env_step = jax.jit(env.step)

    # reset the env
    key, key_reset = jax.random.split(key)
    env_state = env_reset(key_reset)

    # unroll the policy on env. Note that actions are repeated action_repeat
    # times
    pipeline_states = []
    actions = []
    obs = []
    prev_obs = []
    metrics = []
    states = []
    total_reward = 0.0
    for _ in range(episode_length // action_repeat):
        key_sample, key = jax.random.split(key)
        action = policy(env_state.obs, key_sample)[0]
        actions.append(action)
        for _ in range(action_repeat):
            pipeline_states.append(env_state.pipeline_state)
            metrics.append(env_state.metrics)
            obs.append(env_state.obs[:len(env_state.obs)//obs_history_length])
            prev_obs.append(env_state.prev_obs[:len(env_state.prev_obs)//obs_history_length])
            states.append(env_state)
            env_state = env_step(env_state, action)
            total_reward += env_state.reward

    print("Total reward is ", total_reward)

    return actions, pipeline_states, obs, metrics, states, prev_obs


def evaluate_multi_env(params, env_unwrapped: envs.Env,
                       make_policy,
                       episode_length: int,
                       action_repeat: int,
                       key: types.PRNGKey,
                       num_envs: int,
                       obs_history_length: int = 1,
                       deterministic: bool = True):
    if len(params) == 2:
        normalizer_params, policy_params = params
    elif len(params) == 3:
        normalizer_params, policy_params, value_params = params
    eval_env = envs.training.wrap(
        env_unwrapped, episode_length=episode_length,
        action_repeat=action_repeat,
        obs_history_length=obs_history_length)
    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic),
        num_eval_envs=num_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=key)
    metrics = evaluator.run_evaluation((normalizer_params, policy_params),
                                       training_metrics={})
    return metrics

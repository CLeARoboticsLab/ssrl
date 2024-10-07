import streamlit.components.v1 as components
from brax import envs
from brax.base import State
from brax.io import html
from typing import List
from brax.training import types
from brax.envs.wrappers.training import FrameStackWrapper
import jax


def render(env: envs.Env, pipeline_states: List[State], height: int = 500):
    # replace dt in sys with env.dt since env.dt = n_frames * env.sys.dt, but
    # render uses env.sys.dt for rendering (and our timesteps are env.dt)
    render = html.render(env.sys.replace(dt=env.dt), pipeline_states,
                         height=height)
    components.html(render, height=height)


def eval_for_render(policy_params, env_unwrapped: envs.Env,
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
    policy = make_policy(policy_params, deterministic=deterministic)
    env = FrameStackWrapper(env_unwrapped, obs_history_length=obs_history_length)
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
    for _ in range(int(episode_length / action_repeat)):
        key_sample, key = jax.random.split(key)
        actions = policy(env_state.obs, key_sample)[0]
        for _ in range(action_repeat):
            pipeline_states.append(env_state.pipeline_state)
            env_state = env_step(env_state, actions)

    return pipeline_states

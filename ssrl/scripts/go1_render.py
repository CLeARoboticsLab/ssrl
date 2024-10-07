"""Run this script with:
python -m streamlit run go1_render.py"""

import dill
import hydra
from omegaconf import DictConfig
from pathlib import Path
import functools as ft
import streamlit as st


@hydra.main(config_path="configs", config_name="go1")
def render_go1(cfg: DictConfig):

    from brax.envs.go1_deltaxy_pd_slow import Go1DeltaXyPdSlow as Go1
    from brax.io.rendering import render

    # load states to render
    render_name = f"go1_{cfg.render.policy}_render.pkl"
    path = (Path(__file__).parent.parent
            / 'saved_policies'
            / render_name)
    with open(path, 'rb') as f:
        pipeline_states = dill.load(f)

    # create env
    env_fn = ft.partial(Go1, backend='generalized')
    env_fn = add_kwargs_to_fn(env_fn, **cfg.env_eval)
    env = env_fn()

    render_heght = 800
    st.set_page_config(layout="wide")
    render(env, pipeline_states, render_heght)


def add_kwargs_to_fn(partial_fn, **kwargs):
    """add the kwargs to the passed in partial function"""
    for param in kwargs:
        partial_fn.keywords[param] = kwargs[param]
    return partial_fn


if __name__ == '__main__':
    render_go1()

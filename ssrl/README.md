# SSRL

Contains our implementation for Semi-structured Reinforcement Learning (SSRL). Physics simulation and computation of the Lagrangian dynamics are performed using Brax; the code here was forked from the [Brax repository](https://github.com/google/brax).

## Installation

First, if [conda](https://docs.anaconda.com/miniconda/miniconda-install/) / [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) is not already installed, install one of them. mamba is recommended and can be installed with the following:

```sh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Here we create a conda environment, install JAX for CUDA, and then install the package. If CUDA is not available, follow the CPU installation instructions [here](https://github.com/jax-ml/jax). Starting in the `ssrl/ssrl` directory:

```sh
conda env create -n ssrl --file environment.yml
conda activate ssrl
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .
```

## Go1 Simulated Training

Note: configuration is found in `scripts/configs/go1.yaml`.
Note: Wandb logging can be turned on/off in the `wandb` section of `scripts/configs/go1.yaml`.

Train using SSRL with default settings:

```sh
python scripts/go1_train.py
```

Train using black-box models:

```sh
python scripts/go1_train.py ssrl_dynamics_fn=mbpo
```

Training with a single-step loss:

```sh
python scripts/go1_train.py ssrl.model_loss_horizon=1
```

Training can be rendered to [wandb](https://wandb.ai/) by setting `wandb.entity` to your organization, `wandb.log_ssrl` to `true` and `render_epoch_interval` to the number of epochs between renders:

```sh
python scripts/go1_train.py wandb.entity=<YOUR_ORG_HERE> wandb.log_ssrl=true render_epoch_interval=10
```

![SSRL Simulation](../media/sim.gif)

## Simulated RL Benchmarks

Run SSRL on a standard RL benchmark with the following command, replacing `<env_name_here>` with `ant2`, `hopper2`, or `walker2d2`:

```sh
python scripts/rl_benchmarks.py env=<env_name_here>
```

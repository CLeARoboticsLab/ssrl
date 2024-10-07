# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""setup.py for Brax.

Install for development:

  pip intall -e .
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name="ssrl",
    version="0.1.0",
    description=("Learning to Walk from Three Minutes of Real-World Data with Semi-structured Dynamics Models"),
    author="Jacob Levy",
    author_email="jake.levy@utexas.edu",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CLeARoboticsLab/ssrl",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    scripts=["bin/learn"],
    install_requires=[
        "absl-py",
        "dataclasses; python_version < '3.7'",
        # TODO: remove dm_env after dropping legacy v1 code
        "dm_env",
        "etils",
        "flask",
        "flask_cors",
        "flax",
        # TODO: remove grpcio and gym after dropping legacy v1 code
        "grpcio",
        "gym",
        "jax>=0.4.6",
        "jaxlib>=0.4.6",
        "jaxopt",
        "jinja2",
        "mujoco",
        "numpy",
        "optax",
        "Pillow",
        "pytinyrenderer",
        "scipy",
        "tensorboardX",
        "trimesh==3.9.35",
        "typing-extensions",
    ],
    extras_require={
        "develop": ["pytest", "transforms3d"],
    }
)

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integrations tests for the LLVM CompilerGym environments."""
import os

import gym
import pytest

import compiler_gym  # noqa Register environments
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.common"]


def test_default_cBench_dataset_require(tmpwd, temporary_environ):
    """Test that cBench is downloaded."""
    del temporary_environ

    os.environ["COMPILER_GYM_SITE_DATA"] = str(tmpwd / "site_data")
    env = gym.make("llvm-v0")
    assert not env.benchmarks, "Sanity check"

    # Datasaet is downloaded.
    assert env.require_dataset("cBench-v0")
    assert env.benchmarks

    # Dataset is already downloaded.
    assert not env.require_dataset("cBench-v0")


def test_default_cBench_on_reset(tmpwd, temporary_environ):
    """Test that cBench is downloaded by default when no benchmarks are available."""
    del temporary_environ

    os.environ["COMPILER_GYM_SITE_DATA"] = str(tmpwd / "site_data")
    env = gym.make("llvm-v0")
    assert not env.benchmarks, "Sanity check"

    env.reset()
    assert env.benchmarks
    assert env.benchmark.startswith("benchmark://cBench-v0/")


@pytest.mark.parametrize("benchmark_name", ["benchmark://npb-v0/1", "npb-v0/1"])
def test_dataset_required(tmpwd, temporary_environ, benchmark_name):
    """Test that the required dataset is downlaoded when a benchmark is specified."""
    del temporary_environ

    os.environ["COMPILER_GYM_SITE_DATA"] = str(tmpwd / "site_data")
    env = gym.make("llvm-v0")

    env.reset(benchmark=benchmark_name)

    assert env.benchmarks
    assert env.benchmark.startswith("benchmark://npb-v0/")


if __name__ == "__main__":
    main()

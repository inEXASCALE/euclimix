"""Test configuration and GPU availability checks."""

import os

import pytest
import torch


def pytest_sessionstart(session):
    if os.environ.get("EUCLIMIX_REQUIRE_CUDA") == "1" and not torch.cuda.is_available():
        pytest.exit("EUCLIMIX_REQUIRE_CUDA=1 but no CUDA GPU is available", returncode=1)


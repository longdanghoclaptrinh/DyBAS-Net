"""Keep small synthetic verification workloads fast and deterministic on CPU."""

import torch


def pytest_sessionstart(session):
    torch.set_num_threads(1)

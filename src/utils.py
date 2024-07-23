# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
import random
import typing as tp
import collections
from pathlib import Path
import os


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.noise_upper_limit = high - self.loc
        self.noise_lower_limit = low - self.loc

    def _clamp(self, x) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

    def rsample(self, clip=None, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        noise = eps * self.scale
        if clip is not None:
            noise = torch.clamp(noise, -clip, clip)
        noise = torch.clamp(noise, min=self.noise_lower_limit, max=self.noise_upper_limit)
        x = self.loc + noise
        return x


def soft_update_params(net: torch.nn.Module, target_net: torch.nn.Module, tau: float) -> None:
    """
    Perform a soft update of the parameters of two neural networks.

    Args:
        net (torch.nn.Module): A neural network object, typically the source network whose parameters will be used to update the target network.
        target_net (torch.nn.Module): A neural network object, typically the target network whose parameters will be updated based on the source network.
        tau (float): A scalar value between 0 and 1 that controls the degree of interpolation between the source and target network parameters.
    """
    tau = float(np.clip(tau, 0, 1))
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed: int) -> None:
    """
    Set the same seed on all the libraries
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class eval_mode:
    def __init__(self, *models) -> None:
        self.models = models
        self.prev_states: tp.List[bool] = []

    def __enter__(self) -> None:
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args: tp.Any) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)


class Averager:
    """
    A utility class for computing the average of a set of values over time.
    """

    def __init__(self) -> None:
        self.averages = collections.defaultdict(lambda: 0.0)
        self.counters = collections.defaultdict(lambda: 0.0)

    def update(self, values: tp.Dict[str, float]) -> None:
        """
        Update the averages and counters for a given set of values.

        Args:
            values (dict): A dictionary containing the values and their corresponding values.
        """
        for k in values.keys():
            self.counters[k] += 1
            self.averages[k] = self.averages[k] + (values[k] - self.averages[k]) / self.counters[k]

    def reset(self) -> None:
        """
        Reset the averages and counters to their initial state.
        """
        self.averages = collections.defaultdict(lambda: 0.0)
        self.counters = collections.defaultdict(lambda: 0.0)

    def __str__(self):
        return dict.__repr__(self.averages).__str__()


class MetricLogger:
    """
    A utility class for logging metrics from a dictionary to a CSV file.
    The class automatically infers the keys the first time `dump` is called.

    Attention: we assume that the dictionary to log always contains the same keys.
    """

    def __init__(self, output_file: str) -> None:
        """
        Initialize the MetricLogger object.

        Args:
            output_file (str): The path where the CSV file will be saved.
        """
        self.keys = None  # computed the first time update is called
        self.output_path = Path(output_file)

    def dump(self, metrics: tp.Dict[str, float]) -> None:
        """
        Write the metrics to the CSV file.

        Args:
            metrics (dict): A dictionary containing the metrics and their values.
        """
        if self.keys is None:
            self.keys = list(sorted(metrics.keys()))
            with open(self.output_path, "w") as fp:
                line = ",".join(self.keys) + os.linesep
                fp.write(line)
        with open(self.output_path, "a+") as fp:
            line = ",".join([str(float(metrics[k])) for k in self.keys]) + os.linesep
            fp.write(line)

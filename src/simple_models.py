# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# TODO update with ref to realease code utils
from .utils import TruncatedNormal


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, DenseParallel):
        gain = nn.init.calculate_gain("relu")
        parallel_orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()
    # else:
    #     # TODO remove this verbose print 
    #     print("Not applying custom init to layer {}".format(m))


# Initialization for parallel layers
def parallel_orthogonal_(tensor, gain=1):
    if tensor.ndimension() == 2:
        tensor = nn.init.orthogonal_(tensor, gain=gain)
        return tensor
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 3 or more dimensions are supported")
    n_parallel = tensor.size(0)
    rows = tensor.size(1)
    cols = tensor.numel() // n_parallel // rows
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)

    qs = []
    for flat_tensor in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat_tensor.t_()

        # Compute the qr factorization
        q, r = torch.linalg.qr(flat_tensor)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph

        if rows < cols:
            q.t_()
        qs.append(q)

    qs = torch.stack(qs, dim=0)
    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)
        tensor.mul_(gain)
    return tensor


# NOTE removed hidden_sn, inplace_relu, output_mod
def relu_parallel_mlp(
    input_dim, hidden_dim, output_dim, n_parallel, hidden_depth, first_parallel=False
):
    if n_parallel is None or n_parallel <= 1:
        if hidden_depth == 0:
            mods = [
                nn.Linear(
                    input_dim,
                    output_dim,
                )
            ]
        else:
            mods = [
                nn.Linear(
                    input_dim,
                    hidden_dim,
                ),
                nn.ReLU(),
            ]

            def make_hidden():
                l = nn.Linear(
                    hidden_dim,
                    hidden_dim,
                )
                return l

    else:
        if hidden_depth == 0:
            mods = [
                DenseParallel(
                    input_dim,
                    output_dim,
                    n_parallel=n_parallel,
                    first_parallel=first_parallel,
                )
            ]
        else:
            mods = [
                DenseParallel(
                    input_dim,
                    hidden_dim,
                    n_parallel=n_parallel,
                    first_parallel=first_parallel,
                ),
                nn.ReLU(),
            ]

            def make_hidden():
                l = DenseParallel(
                    hidden_dim,
                    hidden_dim,
                    n_parallel=n_parallel,
                    first_parallel=False,
                )
                return l

    for _ in range(hidden_depth - 1):
        mods += [make_hidden(), nn.ReLU()]
    if n_parallel is None or n_parallel <= 1:
        mods.append(nn.Linear(hidden_dim, output_dim))
    else:
        mods.append(
            DenseParallel(
                hidden_dim,
                output_dim,
                n_parallel=n_parallel,
                first_parallel=False,
            )
        )
    return nn.Sequential(*mods)


class DenseParallel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_parallel: int,
        first_parallel: bool,
        bias: bool = True,
        device=None,
        dtype=None,
        reset_params=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(DenseParallel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        self.first_parallel = first_parallel
        if n_parallel is None or (n_parallel == 1):
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.weight = nn.Parameter(
                torch.empty((n_parallel, in_features, out_features), **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty((n_parallel, 1, out_features), **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
            if self.bias is None:
                raise NotImplementedError
        if reset_params:
            self.reset_parameters()

    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            weight_list = [m.weight.T for m in module_list]
            target_weight = torch.stack(weight_list, dim=0)
            self.weight.data.copy_(target_weight.data)
            if self.bias:
                bias_list = [ln.bias.unsqueeze(0) for ln in module_list]
                target_bias = torch.stack(bias_list, dim=0)
                self.bias.data.copy_(target_bias.data)

    # TODO why do these layers have their own reset scheme?
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _forward_matmul(self, input):
        return torch.matmul(input, self.weight) + self.bias

    def forward_expand(self, input):
        pre_in = input.expand(self.n_parallel, -1, -1)
        return torch.baddbmm(self.bias, pre_in, self.weight)

    def forward_parallel(self, input):
        return torch.baddbmm(self.bias, input, self.weight)

    def forward(self, input):
        if self.n_parallel is None or (self.n_parallel == 1):
            return F.linear(input, self.weight, self.bias)
        elif self.first_parallel:
            return self.forward_expand(input)
        else:
            return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, n_parallel={}, bias={}".format(
            self.in_features, self.out_features, self.n_parallel, self.bias is not None
        )


class Actor(nn.Module):
    """Base class for actor policies"""

    def __init__(self) -> None:
        super().__init__()
        # This is the network to process observations. Must be implemented by each sub-class
        self.trunk = None

    def get_mean(self, obs):
        return torch.tanh(self.trunk(obs))

    def get_mean_std(self, obs):
        mu = self.get_mean(obs)
        return mu, torch.ones_like(mu) * self.std

    def forward(self, obs):
        mu, std = self.get_mean_std(obs=obs)
        return TruncatedNormal(mu, std)


class FullyParallelActor(Actor):
    # NOTE removed truncate, num_parallel, spectral_norm, compile, compile_kwargs
    def __init__(self, obs_dim, action_dim, hidden_dim, std, num_hidden_layers: int = 2) -> None:
        nn.Module.__init__(
            self,
        )
        self.std = std
        self.trunk = relu_parallel_mlp(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            n_parallel=None,
            hidden_depth=num_hidden_layers,
        )


class FullyParallelCritics(nn.Module):
    # TODO removed spectral_norm, compile, compile_kwargs
    def __init__(
        self, obs_dim, action_dim, hidden_dim, num_parallel: int = 2, num_hidden_layers: int = 1
    ) -> None:
        super().__init__()
        self.input_dim = obs_dim + action_dim
        self.Qs = relu_parallel_mlp(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_parallel=num_parallel,
            hidden_depth=num_hidden_layers,
            first_parallel=True,
        )

    def forward(self, obs, action):
        h = torch.cat([obs, action], dim=-1)
        return self.Qs(h)



class FullyParallelValues(nn.Module):
    def __init__(self, obs_dim, hidden_dim, num_parallel: int = 2, num_hidden_layers: int = 1
    ) -> None:
        super().__init__()
        self.input_dim = obs_dim
        self.Vs = relu_parallel_mlp(input_dim=self.input_dim,
                                    hidden_dim=hidden_dim,
                                    output_dim=1,
                                    n_parallel=num_parallel,
                                    hidden_depth=num_hidden_layers,
                                    first_parallel=True)

    def forward(self, obs):
        return self.Vs(obs)

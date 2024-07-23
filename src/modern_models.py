# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, TypeVar, List
import torch
from torch import nn
import torch.nn.functional as F
import numbers
from .simple_models import Actor, DenseParallel


def modern_mlp(input_dim, hidden_dim, bottleneck_dim, output_dim, n_blocks):
    mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    mods += [
        ModernResidualBlock(
            input_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=hidden_dim,
            n_parallel=None,
        )
        for _ in range(n_blocks)
    ]
    mods.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*mods)


def modern_parallel_mlp(input_dim, hidden_dim, bottleneck_dim, output_dim, n_parallel, n_blocks):
    mods = [DenseParallel(input_dim, hidden_dim, n_parallel, first_parallel=True), nn.ReLU()]
    mods += [
        ModernResidualBlock(
            input_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=hidden_dim,
            n_parallel=n_parallel,
        )
        for _ in range(n_blocks)
    ]
    mods.append(DenseParallel(hidden_dim, output_dim, n_parallel, first_parallel=False))
    return nn.Sequential(*mods)


T_module = TypeVar("T_module", bound=nn.Module)


def parallel_spectral_norm(
    module: T_module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: Optional[int] = None,
    n_parallel: int = 5,
) -> T_module:
    if dim is None:
        dim = 1
    ParallelSpectralNorm.apply(module, name, n_power_iterations, dim, eps, n_parallel)
    return module


class ParallelSpectralNorm:
    def __init__(
        self,
        name: str = "weight",
        n_power_iterations: int = 1,
        dim: int = 1,
        eps: float = 1e-12,
        n_parallel: int = 5,
    ) -> None:
        self.name = name
        self.dim = dim
        assert self.dim != 0, "first dim is for parallel weights"
        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                "got n_power_iterations={}".format(n_power_iterations)
            )
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.n_parallel = n_parallel

    def reshape_weight_to_matrix(self, weight):
        # weight_mat = weight
        # if self.dim != 1:
        #     # permute dim to front after dim 0
        #     weight_mat = weight_mat.permute(0, self.dim,
        #                                     *[d for d in range(1, weight_mat.dim()) if d != self.dim])
        # height = weight_mat.size(1)
        return weight

    def compute_weight(self, module: nn.Module, do_power_iteration: bool) -> torch.Tensor:
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")  # n_par x 1 x m
        v = getattr(module, self.name + "_v")  # n_par x 1 x n
        weight_mat = self.reshape_weight_to_matrix(weight)  # n_par x m x n

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v = F.normalize(
                        torch.matmul(u, weight_mat), dim=-1, eps=self.eps, out=v
                    )  # n_par x 1 x n
                    u = F.normalize(
                        torch.matmul(v, weight_mat.permute(0, 2, 1)), dim=-1, eps=self.eps, out=u
                    )  # n_par x 1 x m
                if self.n_power_iterations > 0:
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.matmul(torch.matmul(u, weight_mat), v.permute(0, 2, 1))  # n_par x 1 x 1
        weight = weight / sigma
        return weight

    def remove(self, module: nn.Module) -> None:
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + "_u")
        delattr(module, self.name + "_v")
        delattr(module, self.name + "_orig")
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module: nn.Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    @staticmethod
    def apply(
        module, name: str, n_power_iterations: int, dim: int, eps: float, n_parallel: int
    ) -> "ParallelSpectralNorm":
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, ParallelSpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = ParallelSpectralNorm(name, n_power_iterations, dim, eps, n_parallel)
        weight = module._parameters[name]
        if isinstance(weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError(
                "The module passed to `SpectralNorm` can't have uninitialized parameters. "
                "Make sure to run the dummy forward before applying spectral normalization"
            )

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            n_par, h, w = weight_mat.size()
            u = F.normalize(weight.new_empty(n_par, 1, h).normal_(0, 1), dim=-1, eps=fn.eps)
            v = F.normalize(weight.new_empty(n_par, 1, w).normal_(0, 1), dim=-1, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(ParallelSpectralNormStateDictHook(fn))
        return fn


class ParallelSpectralNormStateDictHook:
    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata) -> None:
        if "spectral_norm" not in local_metadata:
            local_metadata["spectral_norm"] = {}
        key = self.fn.name + ".version"
        if key in local_metadata["spectral_norm"]:
            raise RuntimeError("Unexpected key in metadata['spectral_norm']: {}".format(key))
        local_metadata["spectral_norm"][key] = self.fn._version


class ModernResidualBlock(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim, n_parallel=None) -> None:
        super(ModernResidualBlock, self).__init__()
        self._id = input_dim
        self._od = output_dim
        self._bd = bottleneck_dim
        self._np = n_parallel

        def make_fc(
            in_features,
            out_features,
        ):
            if self._np is not None and (not self._np == 1):
                l = DenseParallel(
                    in_features=in_features,
                    out_features=out_features,
                    n_parallel=n_parallel,
                    first_parallel=False,
                )
                l = parallel_spectral_norm(l)
            else:
                l = nn.Linear(in_features=in_features, out_features=out_features)
                l = nn.utils.spectral_norm(l)
            return l

        if self._id != self._od:
            self._short = make_fc(self._id, self._od)
        else:
            self._short = None

        res_layers: List[nn.Module] = []

        if n_parallel is not None:
            res_layers += [ParallelLayerNorm([self._id], n_parallel, elementwise_affine=True)]
        else:
            res_layers += [nn.LayerNorm([self._id], elementwise_affine=True)]

        res_layers += [make_fc(self._id, self._bd), nn.ReLU(), make_fc(self._bd, self._od)]
        self._res = nn.Sequential(*res_layers)

    def forward(self, input):
        res_out = self._res(input)
        if self._short is not None:
            id = self._short(input)
        else:
            id = input
        return id + res_out


class ParallelLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        n_parallel,
        eps=1e-5,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(ParallelLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [
                normalized_shape,
            ]
        assert len(normalized_shape) == 1
        self.n_parallel = n_parallel
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            if n_parallel is None or (n_parallel == 1):
                self.weight = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
            else:
                self.weight = nn.Parameter(
                    torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs)
                )
                self.bias = nn.Parameter(
                    torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs)
                )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            if self.elementwise_affine:
                ln_weights = [ln.weight.unsqueeze(0) for ln in module_list]
                ln_biases = [ln.bias.unsqueeze(0) for ln in module_list]
                target_ln_weights = torch.stack(ln_weights, dim=0)
                target_ln_bias = torch.stack(ln_biases, dim=0)
                self.weight.data.copy_(target_ln_weights.data)
                self.bias.data.copy_(target_ln_bias.data)

    def forward(self, input):
        norm_input = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
        if self.elementwise_affine:
            return (norm_input * self.weight) + self.bias
        else:
            return norm_input

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(
            **self.__dict__
        )


class ModernActor(Actor):
    # NOTE removed truncate, num_parallel, spectral_norm, compile, compile_kwargs, initial_tanh
    def __init__(self, obs_dim, action_dim, hidden_dim, std, num_modern_blocks: int = 1) -> None:
        nn.Module.__init__(
            self,
        )
        self.std = std
        self.input_dim = obs_dim
        self.trunk = modern_mlp(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=hidden_dim,
            output_dim=action_dim,
            n_blocks=num_modern_blocks,
        )


class ModernParallelCritics(nn.Module):
    # NOTE removed spectral_norm, compile, compile_kwargs, initial_tanh
    def __init__(
        self, obs_dim, action_dim, hidden_dim, num_parallel: int = 2, num_modern_blocks: int = 1
    ) -> None:
        super().__init__()
        self.input_dim = obs_dim + action_dim
        self.Qs = modern_parallel_mlp(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=hidden_dim,
            output_dim=1,
            n_parallel=num_parallel,
            n_blocks=num_modern_blocks,
        )

    def forward(self, obs, action):
        h = torch.cat([obs, action], dim=-1)
        return self.Qs(h)


class ModernParallelValues(nn.Module):
    def __init__(self, obs_dim, hidden_dim, num_parallel: int = 2, num_modern_blocks: int = 1) -> None:
        super().__init__()
        self.input_dim = obs_dim
        self.Vs = modern_parallel_mlp(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=hidden_dim,
            output_dim=1,
            n_parallel=num_parallel,
            n_blocks=num_modern_blocks,
        )
            
    def forward(self, obs):
        return self.Vs(obs)

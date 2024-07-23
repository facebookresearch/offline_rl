# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=unused-import
import pdb
import copy
import logging
import dataclasses
import typing as tp
import torch
import numpy as np
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore

from ..simple_models import FullyParallelValues
from ..modern_models import ModernParallelValues

# from .td3aw import TD3awAgentConfig
from ..agent.td3aw import TD3awAgentConfig, TD3awAgent
from ..agent.td3 import ArchiConfig


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ValueArchiConfig:
    hidden_dim: tp.Optional[int] = 1024 
    model: str = 'simple'
    num_parallel: int = 1
    num_hidden_layers: int = 2
    num_modern_blocks: int = 1 # ignored when model="simple"


@dataclasses.dataclass
class IQLArchiConfig(ArchiConfig):
    value: ValueArchiConfig = dataclasses.field(default_factory=ValueArchiConfig)


@dataclasses.dataclass
class IQLAgentConfig(TD3awAgentConfig):
    # @package agent
    _target_: str = "src.agent.iql.IQLAgent"
    name: str = "iql"
    expectile: float = 0.7 
    archi: IQLArchiConfig = dataclasses.field(default_factory=IQLArchiConfig)


cs = ConfigStore.instance()
cs.store(group="agent", name="iql", node=IQLAgentConfig)


def quantile_regression_loss(diff, expectile):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return (weight * (diff ** 2)).mean()


class IQLAgent(TD3awAgent):

    # pylint: disable=unused-argument
    def __init__(self,
                 **kwargs: tp.Any
                 ):
        super().__init__(**kwargs)
        cfg_fields = {field.name for field in dataclasses.fields(IQLAgentConfig)}
        # lets curate the fields others will need to be ported in time
        kwargs = {x: y for x, y in kwargs.items() if x in cfg_fields}
        cfg = IQLAgentConfig(**kwargs)
        self.cfg = cfg
        opt = self.cfg.optim
        value_arch = self.cfg.archi.value
        # create value model
        if value_arch.model == 'simple':
            value_net = FullyParallelValues(self.cfg.obs_shape[0], 
                                          hidden_dim = value_arch.hidden_dim, 
                                          num_parallel = value_arch.num_parallel, 
                                          num_hidden_layers = value_arch.num_hidden_layers)
            
        elif value_arch.model == 'modern':
            value_net = ModernParallelValues(self.cfg.obs_shape[0], 
                                           hidden_dim=value_arch.hidden_dim,
                                           num_parallel = value_arch.num_parallel, 
                                           num_modern_blocks = value_arch.num_modern_blocks)
        else:
            raise NotImplementedError("value model can be only simple or modern")
        
        self.nets["value"] = value_net
        self.nets["value"].to(self.cfg.device)
        self.optims["value"] = torch.optim.Adam(self.nets["value"].parameters(), lr=opt.lr)
    
    def update_critic(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
            discount: torch.Tensor,
            next_obs: torch.Tensor
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        opt = self.cfg.optim
        # update value
        # compute target critic
        with torch.no_grad():
            Qs = self.targets["critic"](obs, action)  # num_parallel x batch x 1
            Q, mean_Q, unc_Q = self.get_critics_targets_uncertainty(Qs)
        
        V = self.nets["value"](obs) # num_parallel x batch x 1
        value_loss = quantile_regression_loss(Q - V, expectile=self.cfg.expectile)

        # optimize value
        self.optims["value"].zero_grad(set_to_none=True)
        value_loss.backward()
        self.optims["value"].step()

        # update critic
        # compute target critic
        with torch.no_grad():
            next_V = self.nets["value"](next_obs)
            target_Q = reward + discount * next_V
            expanded_targets = target_Q.expand(self.cfg.archi.critic.num_parallel, -1, -1)
        
        # current critic
        Qs = self.nets["critic"](obs, action)
        critic_loss: tp.Any = 0.5 * self.cfg.archi.critic.num_parallel * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.optims["critic"].zero_grad(set_to_none=True)
        critic_loss.backward()
        self.optims["critic"].step()

        with torch.no_grad():
            metrics['target_Q'] = target_Q.mean().item()
            metrics['Q1'] = Qs[0].mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['value_loss'] = value_loss.item()
            metrics['V'] = V.mean().item()
        
        return metrics
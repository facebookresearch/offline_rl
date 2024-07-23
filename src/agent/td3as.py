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

from ..agent.td3aw import TD3awAgentConfig, TD3awAgent
from ..agent.td3 import ArchiConfig
from ..utils import soft_update_params



logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TD3asAgentConfig(TD3awAgentConfig):
    # @package agent
    _target_: str = "src.agent.td3as.TD3asAgent"
    name: str = "td3as"


cs = ConfigStore.instance()
cs.store(group="agent", name="td3as", node=TD3asAgentConfig)


class TD3asAgent(TD3awAgent):

    # pylint: disable=unused-argument
    def __init__(self,
                 **kwargs: tp.Any
                 ):
        super().__init__(**kwargs)
        cfg_fields = {field.name for field in dataclasses.fields(TD3asAgentConfig)}
        # lets curate the fields others will need to be ported in time
        kwargs = {x: y for x, y in kwargs.items() if x in cfg_fields}
        cfg = TD3asAgentConfig(**kwargs)
    
    def update_actor(self, obs: torch.Tensor, dataset_action: torch.Tensor) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        opt = self.cfg.optim
        dist = self.nets["actor"](obs)
        dataset_log_probs = dist.log_prob(dataset_action).mean(dim=-1, keepdim=True) # average over action dimension
        likelihood_loss = -1 * dataset_log_probs # NOTE: wrong KL # given that std = 0.2 - it effective multiplis by 50 the MSE!

        actor_loss = likelihood_loss.mean()
        self.optims["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optims["actor"].step()

        with torch.no_grad():
            metrics['actor_loss'] = actor_loss.item()
        return metrics

    def update(self, replay_loader, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # sample
        uniform_batch, uniform_tree_idx = replay_loader.sample_with_tree_idx(
            batch_size=self.cfg.batch_size, prioritized=False)

        prioritized_batch, prioritized_tree_idx = replay_loader.sample_with_tree_idx(
            batch_size=self.cfg.batch_size, prioritized=True)

        # update critic
        obs, action, next_obs, reward, discount = uniform_batch
        metrics.update(self.update_critic(obs=obs, action=action, reward=reward, discount=discount, next_obs=next_obs))

        # update_actor
        prioritized_obs, prioritized_action, _ , _, _ = prioritized_batch
        metrics.update(self.update_actor(obs = prioritized_obs, dataset_action = prioritized_action))

        # update critic target
        with torch.no_grad():
            soft_update_params(self.nets["critic"], self.targets["critic"], self.cfg.optim.critic_target_tau)

        all_tree_idx = np.concatenate([prioritized_tree_idx, uniform_tree_idx], axis=0)
        all_obs = torch.concat([prioritized_obs, obs], dim=0)
        all_actions = torch.concat([prioritized_action, action], dim=0)
        dist = self.nets["actor"](all_obs)
        dataset_advantage = self.estimate_advantage(dist=dist, obs=all_obs, dataset_action=all_actions)
        scores = dataset_advantage.cpu().numpy() / self.cfg.awac.beta
        # breakpoint()
        replay_loader.update_probabilities(
            unnormalized_probabilities=scores.squeeze(), 
            tree_indexes=all_tree_idx.squeeze(),
            pop_values=False)

        return metrics


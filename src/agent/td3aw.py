# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
import typing as tp
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf

# TODO port the code and update
from ..agent.td3 import TD3AgentConfig, TD3Agent


@dataclasses.dataclass
class AwacConfig:
    beta: float = 1.0
    num_value_samples: int = 1
    max_clip: tp.Optional[float] = None


@dataclasses.dataclass
class TD3awAgentConfig(TD3AgentConfig):
    # @package agent
    _target_: str = "src.agent.td3aw.TD3awAgent"
    name: str = "td3aw"
    awac: AwacConfig = dataclasses.field(default_factory=AwacConfig)



cs = ConfigStore.instance()
cs.store(group="agent", name="td3aw", node=TD3awAgentConfig)


class TD3awAgent(TD3Agent):

    # pylint: disable=unused-argument
    def __init__(self, **kwargs: tp.Any):
        super().__init__(**kwargs)
        # make config
        cfg_fields = {field.name for field in dataclasses.fields(TD3awAgentConfig)}
        kwargs = {x: y for x, y in kwargs.items() if x in cfg_fields}
        self.cfg = TD3awAgentConfig(**kwargs)
    
    # NOTE: we always use the online critic 
    def estimate_advantage(self, dist, obs, dataset_action):
        with torch.no_grad():
            batch_size = obs.shape[0]
            actions = dist.rsample(sample_shape=(self.cfg.awac.num_value_samples,))
            action_dims = actions.shape
            obs_exp = obs.expand(self.cfg.awac.num_value_samples, -1, -1)
            actions_flat = actions.view(self.cfg.awac.num_value_samples * batch_size, action_dims[-1])
            obs_flat = torch.flatten(obs_exp, start_dim=0, end_dim=-2)
            
            Qs_flat = self.nets["critic"](obs_flat, actions_flat)
            Q_flat, _, _  = self.get_critics_targets_uncertainty(qs=Qs_flat)
            Q = Q_flat.view(self.cfg.awac.num_value_samples, batch_size, 1)
            value = Q.mean(0)

            dataset_Qs = self.nets["critic"](obs, dataset_action)
            dataset_Q = dataset_Qs.mean(0) # no pessimism for dataset action
            dataset_advantage = dataset_Q - value

        return dataset_advantage

    def awac_process_score(self, score: torch.Tensor) -> torch.Tensor:
        if self.cfg.awac.max_clip is not None:
            score = score.clamp(max=self.cfg.awac.max_clip)
        scaled_scores = score/self.cfg.awac.beta
        weights = torch.softmax(scaled_scores, dim=0)
        return weights
            

    def update_actor(self, obs: torch.Tensor, dataset_action: torch.Tensor) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        opt = self.cfg.optim
        dist = self.nets["actor"](obs)
        dataset_advantage = self.estimate_advantage(dist=dist, obs=obs, dataset_action=dataset_action)
        
        dataset_advantage_weight = self.awac_process_score(score=dataset_advantage).detach()
        dataset_log_probs = dist.log_prob(dataset_action).mean(dim=-1, keepdim=True) # average over action dimension
        likelihood_loss = -1 * dataset_log_probs # NOTE: wrong KL # given that std = 0.2 - it effective multiplis by 50 the MSE!

        actor_loss = (dataset_advantage_weight * likelihood_loss).mean() # NOTE w/ softmax might be intuitive to have sum - likely does not play a major difference as ADAM is rescaling the gradients
        
        self.optims["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optims["actor"].step()

        with torch.no_grad():
            metrics['policy_log_likelihood_loss'] = likelihood_loss.mean().item()
            metrics['max_weight'] = dataset_advantage_weight.max().item()
            metrics['weight_std'] = torch.std(dataset_advantage_weight).item()
            metrics['actor_loss'] = actor_loss.item()
        return metrics
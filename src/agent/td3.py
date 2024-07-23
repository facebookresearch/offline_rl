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

from ..simple_models import FullyParallelActor, FullyParallelCritics
from ..modern_models import ModernActor, ModernParallelCritics
from ..simple_models import weight_init
from ..buffer import InMemoryBuffer
from ..utils import soft_update_params


@dataclasses.dataclass
class OptimConfig:
    lr: float = 1e-4
    critic_target_tau: float = 0.005
    std: float = 0.2
    std_clip: float = 0.3
    pessimism_penalty: float = 0.5
    actor_bc_coeff: float = 0.0 

    
@dataclasses.dataclass
class ActorArchiConfig:
    hidden_dim: int = 1024
    model: str = 'simple'  # ["simple", "modern"]
    num_hidden_layers: int = 2
    num_modern_blocks: int = 1  # ignored when model="simple"
    eval_actor_samples: int = 1
    

@dataclasses.dataclass
class CriticArchiConfig:
    hidden_dim: int = 256
    model: str = 'simple'  # ["simple", "modern"]
    num_parallel: int = 2
    num_hidden_layers: int = 2
    num_modern_blocks: int = 1  # ignored when model="simple"


@dataclasses.dataclass
class ArchiConfig:
    actor: ActorArchiConfig = dataclasses.field(default_factory=ActorArchiConfig)
    critic: CriticArchiConfig = dataclasses.field(default_factory=CriticArchiConfig)


@dataclasses.dataclass
class TD3AgentConfig:
    # @package agent
    _target_: str = "src.agent.td3.TD3Agent"
    name: str = "td3"
    batch_size: int = 1024
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING 
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING
    device: str = omegaconf.II("device")
    archi: ArchiConfig = dataclasses.field(default_factory=ArchiConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)

    # TODO remove (used only for testing with imitation.train)
    obs_type: str = omegaconf.MISSING
    num_expl_steps: int = omegaconf.MISSING
    discrete_action: bool = omegaconf.MISSING
    goal_space: tp.Optional[str] = omegaconf.II("goal_space")



cs = ConfigStore.instance()
cs.store(group="agent", name="td3", node=TD3AgentConfig)


class TD3Agent:

    # pylint: disable=unused-argument
    def __init__(self, **kwargs: tp.Any):
        # make config
        cfg_fields = {field.name for field in dataclasses.fields(TD3AgentConfig)}
        kwargs = {x: y for x, y in kwargs.items() if x in cfg_fields}
        self.cfg = TD3AgentConfig(**kwargs)
        actor_arch = self.cfg.archi.actor
        critic_arch = self.cfg.archi.critic
        self.pessimism_penalty_scaling = critic_arch.num_parallel**2 - critic_arch.num_parallel
        self.eval_actor_samples = self.cfg.archi.actor.eval_actor_samples
        # create actor model
        if actor_arch.model == 'simple':
            actor = FullyParallelActor(self.cfg.obs_shape[0], self.cfg.action_shape[0],
                                       hidden_dim=actor_arch.hidden_dim,
                                       std=self.cfg.optim.std,
                                       num_hidden_layers = actor_arch.num_hidden_layers)
        elif actor_arch.model == 'modern':
            actor = ModernActor(self.cfg.obs_shape[0], self.cfg.action_shape[0],
                                hidden_dim=actor_arch.hidden_dim,
                                std=self.cfg.optim.std,
                                num_modern_blocks = actor_arch.num_modern_blocks)
        else:
            raise NotImplementedError("actor model can be only simple or modern")
        # create critic model
        if critic_arch.model == 'simple':
            critic = FullyParallelCritics(self.cfg.obs_shape[0], self.cfg.action_shape[0], 
                                          hidden_dim = critic_arch.hidden_dim, 
                                          num_parallel = critic_arch.num_parallel, 
                                          num_hidden_layers = critic_arch.num_hidden_layers)
            
        elif critic_arch.model == 'modern':
            critic = ModernParallelCritics(self.cfg.obs_shape[0], self.cfg.action_shape[0], 
                                           hidden_dim=critic_arch.hidden_dim,
                                           num_parallel = critic_arch.num_parallel, 
                                           num_modern_blocks = critic_arch.num_modern_blocks)
        else:
            raise NotImplementedError("critic model can be only simple or modern")

        # initialize networks and targets
        self.nets = dict(actor=actor, critic=critic)
        for net in self.nets.values():
            net.apply(weight_init)
        print(self.nets["actor"])
        print(self.nets["critic"])
        
        self.nets = {x: y.to(self.cfg.device) for x, y in self.nets.items()}
        self.targets = {"critic": copy.deepcopy(self.nets["critic"])}

        self.train()
        for net in self.targets.values():
            net.train()

        # optimizers
        self.optims = dict(actor=torch.optim.Adam(self.nets["actor"].parameters(), lr=self.cfg.optim.lr),
                           critic=torch.optim.Adam(self.nets["critic"].parameters(), lr=self.cfg.optim.lr))

    def train(self, training: bool = True) -> None:
        self.training = training
        for net in self.nets.values():
            net.train(training)

    # NOTE removed unused argument transform ='linear'
    # NOTE removed unused return value q_unc
    # NOTE moved to a class method
    def get_critics_targets_uncertainty(self, qs):
        q_mean = qs.mean(0)
        if self.cfg.archi.critic.num_parallel == 2:
            q_unc = torch.abs(qs[0, ...] - qs[1, ...])
        elif self.cfg.archi.critic.num_parallel > 2:
            qs_uns = qs.unsqueeze(0) # 1 x n_parallel x ...
            qs_uns2 = qs.unsqueeze(1) # n_parallel x 1 x ...
            qs_diffs = torch.abs(qs_uns - qs_uns2)
            q_unc = qs_diffs.sum(dim=(0, 1)) / self.pessimism_penalty_scaling
        return q_mean - self.cfg.optim.pessimism_penalty * q_unc, q_mean, q_unc

    # TODO (edoardo): refactor for compatibility w/ parallel actors
    # TODO (andrea): what does it mean?
    def batch_act(self, obs: torch.Tensor, step: int, eval_mode: bool) -> torch.Tensor:
        dist = self.nets["actor"](obs)
        if eval_mode:
            if self.eval_actor_samples > 1:
                bs = obs.shape[0]
                actions = dist.rsample(sample_shape=(self.eval_actor_samples,))
                action_dims = actions.shape
                obs_exp = obs.expand(self.eval_actor_samples, -1, -1)
                actions_flat = actions.view(self.eval_actor_samples*bs, action_dims[-1])
                obs_flat = torch.flatten(obs_exp, start_dim=0,  end_dim=-2)
                
                Qs_flat = self.nets["critic"](obs_flat, actions_flat)
                Q_flat, _, _  = self.get_critics_targets_uncertainty(qs=Qs_flat)
                Q = Q_flat.view(self.eval_actor_samples, bs, 1)
                Qs_argmax = torch.argmax(Q, dim=0, keepdim=True).expand_as(actions)
                
                action = torch.gather(actions, dim=0, index=Qs_argmax)[0]
            else:
                action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action

    def act(self, obs, step: int, eval_mode: bool) -> tp.Any:
        obs = torch.as_tensor(obs, device=self.cfg.device, dtype=torch.float32).unsqueeze(0)  # type: ignore
        action = self.batch_act(obs=obs, step=step, eval_mode=eval_mode)
        return action.cpu().numpy()[0]
    
    def update_critic(self,
                      obs: torch.Tensor,
                      action: torch.Tensor,
                      reward: torch.Tensor,
                      discount: torch.Tensor,
                      next_obs: torch.Tensor) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # compute target critic
        with torch.no_grad():
            dist = self.nets["actor"](next_obs)
            next_action = dist.rsample(clip=self.cfg.optim.std_clip)
            next_Qs = self.targets["critic"](next_obs, next_action)  # num_parallel x batch x 1
            next_V, mean_Q, unc_Q = self.get_critics_targets_uncertainty(next_Qs)
            target_Q = reward + discount * next_V
            expanded_targets = target_Q.expand(self.cfg.archi.critic.num_parallel, -1, -1)
        # compute critic loss
        Qs = self.nets["critic"](obs, action)
        critic_loss: tp.Any = 0.5*self.cfg.archi.critic.num_parallel*F.mse_loss(Qs, expanded_targets)
        with torch.no_grad():
            metrics['target_Q'] = target_Q.mean().item()
            metrics['Q1'] = Qs[0].mean().item()
            metrics['mean_next_Q'] = mean_Q.mean().item()
            metrics['unc_Q'] = unc_Q.mean().item()
            metrics['critic_loss'] = critic_loss.item()
        # optimize critic
        self.optims["critic"].zero_grad(set_to_none=True)
        critic_loss.backward()
        self.optims["critic"].step()
        return metrics                                                         
    
    def update_actor(self, obs: torch.Tensor, dataset_action: torch.Tensor) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        dist = self.nets["actor"](obs)
        action = dist.rsample()
        
        # compute Q loss
        Qs = self.nets["critic"](obs, action)
        Q, _, _  = self.get_critics_targets_uncertainty(qs=Qs)
        actor_loss = -Q.mean()

        # compute bc loss
        if self.cfg.optim.actor_bc_coeff > 0:
            bc_error = F.mse_loss(action, dataset_action)
            bc_loss = self.cfg.optim.actor_bc_coeff * Q.abs().mean().detach() * bc_error
            actor_loss = actor_loss + bc_loss
            with torch.no_grad():
                metrics['bc_loss'] = bc_loss.item()

        with torch.no_grad():
            metrics['actor_loss'] = actor_loss.item()

        # optimize actor
        self.optims["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optims["actor"].step()
        return metrics

    # TODO update based on calling interface
    def update(self, replay_loader: tp.Any, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        if isinstance(replay_loader, dict):
            replay_loader = replay_loader["train"]
        batch = replay_loader.sample(self.cfg.batch_size)

        obs, action, next_obs, reward, discount = batch
            
        # update critic
        metrics.update(self.update_critic(obs=obs, action=action, reward=reward, discount=discount, next_obs=next_obs))
        # update actor
        metrics.update(self.update_actor(obs, dataset_action=action))
        # update critic target
        with torch.no_grad():
            soft_update_params(self.nets["critic"], self.targets["critic"], self.cfg.optim.critic_target_tau)

        return metrics

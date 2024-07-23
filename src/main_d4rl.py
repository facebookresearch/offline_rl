# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import gym
import d4rl
import dataclasses

import typing as tp
import hydra
from hydra.core.config_store import ConfigStore
import time
from pathlib import Path
import copy
from omegaconf import OmegaConf
import wandb

from . import utils as utils
from .buffer import InMemoryBuffer
from .log_prioritized_buffer import LogPrioritizedBuffer
from . import agent as agents

# # # Config # # #
@dataclasses.dataclass
class TrainConfig:
    agent: tp.Any
    # misc
    seed: int = 1
    device: str = "cuda"
    # task settings
    task: str = "walker2d-medium-v2"
    discount: float = 0.99
    # eval
    eval_every_steps: int = 10000
    num_eval_episodes: int = 2
    # training
    num_train_steps: int = 2000000
    # evaluation-based sampling
    test_es_values: tp.Tuple[int, ...] = (1, 50)
    # misc
    log_every_updates: int = 1000
    clip_offline_actions: float = np.inf  # TODO: remove?
    # wandb
    use_wandb: bool = False
    wandb_ename: tp.Optional[str] = None
    wandb_gname: tp.Optional[str] = None
    wandb_pname: tp.Optional[str] = None

ConfigStore.instance().store(name="workspace_config", node=TrainConfig)


class Workspace:
    def __init__(self, cfg: TrainConfig) -> None:
        self.output_dir = Path.cwd()
        # Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        print(f"Output dir {self.output_dir}")
        self.cfg = copy.deepcopy(cfg)
        del cfg

        if not torch.cuda.is_available():
            if self.cfg.device != "cpu":
                print(f"Falling back to cpu as {self.cfg.device} is not available")
                self.cfg.device = "cpu"
                self.cfg.agent.device = "cpu"
        self.device = torch.device(self.cfg.device)

        self.eval_env = gym.make(self.cfg.task)
        self.eval_env.seed(self.cfg.seed)
        self.eval_env.action_space.seed(self.cfg.seed)
        self.eval_env.observation_space.seed(self.cfg.seed)
        utils.set_seed_everywhere(self.cfg.seed)

        self.cfg.agent.obs_shape = (self.eval_env.observation_space.shape[0],)
        self.cfg.agent.action_shape = (self.eval_env.action_space.shape[0],)
        #######################################
        # REMOVE
        self.cfg.agent.obs_type = "states"
        self.cfg.agent.goal_space = None
        self.cfg.agent.num_expl_steps = 1
        self.cfg.agent.discrete_action = False
        #######################################

        self.agent = hydra.utils.instantiate(self.cfg.agent, _recursive_=False)

        if self.cfg.use_wandb:
            exp_name = '_'.join([self.cfg.agent.name, self.cfg.task])
            # fmt: off
            wandb.init(entity=self.cfg.wandb_ename, project=self.cfg.wandb_pname,
                group=self.cfg.agent.name if self.cfg.wandb_gname is None else self.cfg.wandb_gname, name=exp_name,  # mode="disabled",
                config=OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=False))  # type: ignore
            # fmt: off
    
    def eval(self, t) -> None:
        print("---------------------------------------")
        print(f"{utils.bcolors.OKGREEN}Evaluation at {t} time steps{utils.bcolors.ENDC}")
        print(f"Total time passed: {round((time.time()-self.start_time)/60.,2)} min(s)")
        print(f"steps-per-second: {round(t / (time.time()-self.start_time),2)}")
        num_ep = self.cfg.num_eval_episodes
        total_reward = np.zeros(num_ep)
        num_columns = len(self.cfg.test_es_values) #[returns_es{x}, ...] 
        total_reward = np.zeros((num_ep, num_columns), dtype=np.float64)
        for i, num_es in enumerate(self.cfg.test_es_values):
            print(f"--# num ES: {num_es} #--")
            self.agent.eval_actor_samples = num_es
            for ep in range(num_ep):
                state, done = self.eval_env.reset(), False
                while not done:
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(obs=state, step=-1, eval_mode=True)
                    state, reward, done, _ = self.eval_env.step(action)
                    total_reward[ep, i] += reward
            print(f"Average total reward over {num_ep} episodes: {total_reward[:, i].mean():.3f}")
            d4rl_score = self.eval_env.get_normalized_score(total_reward[:, i]) * 100
            print(f"D4RL score: {d4rl_score.mean():.3f}")
        print("---------------------------------------")
        with open(self.output_dir / f"ep_returns.csv", "a") as f:
            if t == 0:
                f.write(",".join(["train_step"] + [f"returns_es{num_es}" for num_es in self.cfg.test_es_values] + ["\n"]))
            time_indexed_returns = np.full((num_ep, 1+total_reward.shape[1]), t, dtype=np.float32)
            time_indexed_returns[:, 1:] = total_reward
            np.savetxt(f, time_indexed_returns, fmt=",".join(["%d"] + ["%.9f" for _ in self.cfg.test_es_values]))
        if self.cfg.use_wandb:
            for i, num_es in enumerate(self.cfg.test_es_values):
                wandb.log({
                    "eval/episode_reward_es{num_es}#std": total_reward[:, i].std(),
                    "eval/episode_reward_es{num_es}": total_reward[:, i].mean(),
                }, step=t)

    def train(self) -> None:
        # load D4RL replay buffer
        replay_loader = InMemoryBuffer(
            state_dim=self.eval_env.observation_space.shape[0],
            action_dim=self.eval_env.action_space.shape[0],
            device=self.device,
            max_size=10,  # automatically inferred by loading D4RL
            max_action=float(self.eval_env.action_space.high[0]),
            normalize_actions=False,
            prioritized=False,
        )
        replay_loader.load_D4RL(
            dataset=d4rl.qlearning_dataset(self.eval_env),
            antmaze=("antmaze" in self.cfg.task),
            clip_action=self.cfg.clip_offline_actions,
        )
        if isinstance(self.agent, agents.TD3asAgent):
            replay_loader = LogPrioritizedBuffer(base_buffer=copy.deepcopy(replay_loader))
        self.start_time = time.time()
        metric_average = utils.Averager()
        metric_logger = utils.MetricLogger(output_file=self.output_dir / "metrics.log")

        for t in range(int(self.cfg.num_train_steps)):
            if not t % self.cfg.eval_every_steps:
                self.eval(t=t)

            # update
            metrics = self.agent.update(replay_loader, t)
            metric_average.update(metrics)

            if not t % self.cfg.log_every_updates:
                print(f"train step {t}: ", metric_average)
                metric_average.averages["train_step"] = t
                metric_logger.dump(metrics=metric_average.averages)
                if self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in metric_average.averages.items()},
                        step=t,
                    )
                metric_average.reset()


@hydra.main(config_path=".", config_name="hydra_plugins_config", version_base=None)
def main(cfg: TrainConfig) -> None:
    workspace = Workspace(cfg)  # type: ignore
    workspace.train()


if __name__ == "__main__":
    main()

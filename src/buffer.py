# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



# Adapted from TD7 Replay Buffer (https://github.com/sfujim/TD7/blob/main/buffer.py)
#
# Original License
#
# MIT License
#
# Copyright (c) 2023 Scott Fujimoto
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import numpy as np
import torch


class InMemoryBuffer:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_size=1e6,
        max_action=1,
        normalize_actions=True,
        prioritized=True,
        discount=0.99
    ):

        max_size = int(max_size)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self._discount = discount

        self.device = device

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.prioritized = prioritized
        if prioritized:
            self.priority = torch.zeros(max_size, device=device)
            self.max_priority = 1

        self.normalize_actions = max_action if normalize_actions else 1

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action / self.normalize_actions
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        if self.prioritized:
            self.priority[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def retrieve(self, indices):
        return (
            torch.tensor(self.state[indices], dtype=torch.float, device=self.device),
            torch.tensor(self.action[indices], dtype=torch.float, device=self.device),
            torch.tensor(self.next_state[indices], dtype=torch.float, device=self.device),
            torch.tensor(self.reward[indices], dtype=torch.float, device=self.device),
            self._discount * torch.tensor(self.not_done[indices], dtype=torch.float, device=self.device),
        )

    def sample(self, batch_size):
        if self.prioritized:
            csum = torch.cumsum(self.priority[: self.size], 0)
            val = torch.rand(size=(batch_size,), device=self.device) * csum[-1]
            self.ind = torch.searchsorted(csum, val).cpu().data.numpy()
        else:
            self.ind = np.random.randint(0, self.size, size=batch_size)
        return self.retrieve(self.ind)

        

    def update_priority(self, priority):
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = float(self.priority[: self.size].max())

    def load_D4RL(self, dataset, antmaze=False, clip_action=np.inf):
        rewards = dataset["rewards"].reshape(-1, 1)
        if antmaze:
            # from IQL paper
            # https://github.com/ikostrikov/implicit_q_learning/blob/09d700248117881a75cb21f0adb95c6c8a694cb2/train_offline.py#L69
            rewards = rewards - 1
        self.state = dataset["observations"]
        self.action = dataset["actions"]
        self.action = np.clip(self.action, -clip_action, clip_action)
        self.next_state = dataset["next_observations"]
        self.reward = rewards
        self.not_done = 1.0 - dataset["terminals"].reshape(-1, 1)
        self.size = self.state.shape[0]
        self.max_size = self.size

        if self.prioritized:
            self.priority = torch.ones(self.size).to(self.device)

    def load_MOOD(self, dataset, load_physics=False, clip_action=np.inf):
        """
        dataset is in hdf5 format and contains trajectories.
        Here we generate transitions.
        """
        state_dim = dataset["observation"].shape[-1]
        action_dim = dataset["action"].shape[-1]
        self.state = dataset["observation"][:, :-1].transpose(2, 0, 1).reshape(state_dim, -1).T
        self.action = dataset["action"][:, 1:].transpose(2, 0, 1).reshape(action_dim, -1).T
        self.action = np.clip(self.action, -clip_action, clip_action)
        self.next_state = dataset["observation"][:, 1:].transpose(2, 0, 1).reshape(state_dim, -1).T
        self.reward = dataset["reward"][:, 1:].ravel().reshape(-1, 1)
        self.not_done = 1 - dataset["terminated"][:, 1:].ravel().reshape(-1, 1)
        self.size = self.state.shape[0]
        self.max_size = self.size

        assert self.state.shape[1] >= 1
        assert self.action.shape[1] >= 1
        assert self.next_state.shape[1] >= 1
        assert self.reward.shape[1] == 1
        assert self.not_done.shape[1] == 1
        assert self.action.shape[0] == self.size and self.next_state.shape[0] == self.size
        assert self.reward.shape[0] == self.size and self.not_done.shape[0] == self.size

        if self.prioritized:
            self.priority = torch.ones(self.size).to(self.device)

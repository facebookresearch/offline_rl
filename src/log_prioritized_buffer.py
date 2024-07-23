# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pdb  # pylint: disable=unused-import
import logging
import typing as tp
import dataclasses
import collections

import numpy as np
import torch
from tqdm import tqdm

from .buffer import InMemoryBuffer
from scipy.special import logsumexp

def logsubtractexp(x1, x2):
    # the maximum exp is now guaranteed to be 1
    return np.log1p(-np.exp(x2 - x1)) + x1


# Segment tree data structure keeping probabilities in log space, where parent node values are the log + add + exp of children node values
class LogSegmentTree:
    # TODO: considering depth num of samples and use gumbel softmax rather than needing logsubtractexp if current implementation is too slow.
    def __init__(self, 
                 size, 
                 #probability_temperature: float = 1.0, # to be subtracted in log space
                 ):
        # self.index = 0
        # self.full = False  # Used to track actual capacity
        self.size = size
        self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
        if size % 2 == 0:
            self.padded_size = self.size
        else:
            self.padded_size = self.size + 1
        self.log_sum_tree_total_unpadded_length = self.tree_start + self.size
        self.log_sum_tree_total_length = self.tree_start + self.padded_size
        self.log_sum_tree = np.zeros((self.log_sum_tree_total_length,), dtype=np.float64)-1e8 # initializing with 0 prob in log space
       
        self.max = 0 #self.log_temp_shift  # Initial max value such that at initialization the RB is filled with 1 for each value
        
    # Updates nodes values from current tree
    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.log_sum_tree[indices] = np.logaddexp(self.log_sum_tree[children_indices[0]],
                                                  self.log_sum_tree[children_indices[1]],)#np.sum(self.log_sum_tree[children_indices], axis=0)

    # Propagates changes up tree given tree indices
    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.log_sum_tree[parent] = np.logaddexp(self.log_sum_tree[left],
                                                 self.log_sum_tree[right])
        if parent != 0:
            self._propagate_index(parent)

    # Updates values given tree indices
    def update(self, indices, values, pop_values=False):
        if pop_values:
            popped_values = self.log_sum_tree[indices]
        self.log_sum_tree[indices] = values # - self.log_temp_shift # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)
        if pop_values:
            return popped_values

    # Updates single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.log_sum_tree[index] = value # - self.log_temp_shift # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)

    # Searches for the location of values in sum tree
    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.log_sum_tree_total_length:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.log_sum_tree_total_unpadded_length - 1)
        left_children_values = self.log_sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values) # Classify which values are in left or right branches
        successor_indices = children_indices[successor_choices.astype(np.int32), np.arange(indices.size)] # Use classification to index into the indices matrix
        successor_values = np.where(successor_choices, 
                                    logsubtractexp(values, left_children_values),
                                    values)
        # breakpoint()
        # print('Nan check')
        # print(np.isnan(logsubtractexp(values, left_children_values)).sum())
        # print(np.isnan(successor_values).sum())
        return self._retrieve(successor_indices, successor_values)

    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.log_sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    def sample_uniform(self, batch_size):
        data_idx = np.random.randint(self.size, size=batch_size)
        tree_idx = data_idx + self.tree_start
        return data_idx, tree_idx #Return values, data indices, tree indices
    
    def sample_segments(self, batch_size,):
        p_log_total = self.log_total()
        random_samples = np.random.uniform(0.0, 1.0, [batch_size]) + np.arange(batch_size)
        log_random_samples = np.log(random_samples)
        log_batch_size = np.log(batch_size)
        log_scaled_random_samples = log_random_samples - log_batch_size + p_log_total # rescale to have samples in (0, 1), and then to match p_total
        # print(np.max(log_scaled_random_samples))
        # print(np.min(log_scaled_random_samples))
        # print(random_samples[0])
        #print('PR')
        values, data_idx, tree_idx = self.find(log_scaled_random_samples)  # Retrieve samples from tree with un-normalised probability
        return data_idx, tree_idx #, values
    
    def sample_segments_gumbel(self, batch_size,):
        p_log_total = self.log_total()
        random_samples = np.random.uniform(0.0, 1.0, [batch_size]) + np.arange(batch_size)
        log_random_samples = np.log(random_samples)
        log_batch_size = np.log(batch_size)
        log_scaled_random_samples = log_random_samples - log_batch_size + p_log_total # rescale to have samples in (0, 1), and then to match p_total
        values, data_idx, tree_idx = self.find(log_scaled_random_samples)  # Retrieve samples from tree with un-normalised probability
        return data_idx, tree_idx #, values
    

    def get_value(self, data_index):
        tree_idx = data_index + self.tree_start
        values = self.log_sum_tree[tree_idx]
        return values
    
    def log_total(self):
        return self.log_sum_tree[0]

class LogPrioritizedBuffer:
    def __init__(self, base_buffer: InMemoryBuffer,
                init_priority_tree_logit=None,
                **kwargs: tp.Any) -> None:
        super().__init__(**kwargs)
        self.base_buffer = base_buffer
        self.init_priority_tree_logit = init_priority_tree_logit
        
        self._make_priority_tree()
        

    def _make_priority_tree(self,):
        print('Making prioritized buffer')
        self.priority_tree = LogSegmentTree(size=self.base_buffer.max_size)
        if self.init_priority_tree_logit is None:
            self.init_priority_tree = self.priority_tree.max
        else:
            self.init_priority_tree = self.init_priority_tree_logit
        indices = np.arange(self.base_buffer.max_size) + self.priority_tree.tree_start
        self.priority_tree.update(indices, self.init_priority_tree)
        # for _ in tqdm(range(self.base_buffer.max_size)):
        #     self.priority_tree.append(value=self.init_priority_tree)

    def _get_uniform_sample_index(self, batch_size):
        return self.priority_tree.sample_uniform(batch_size=batch_size)
    
    def _get_prioritized_sample_index(self, batch_size,):
        return self.priority_tree.sample_segments(batch_size=batch_size)
            
    def sample_with_tree_idx(self, batch_size, prioritized: bool = False):

        if prioritized:
            data_idx, tree_idx = self._get_prioritized_sample_index(batch_size=batch_size)
        else:
            data_idx, tree_idx = self._get_uniform_sample_index(batch_size=batch_size)
        data_batch = self.base_buffer.retrieve(data_idx)
        return data_batch, tree_idx

    def update_probabilities(self, unnormalized_probabilities, tree_indexes, pop_values=False): 
        self.priority_tree.update(indices=tree_indexes, values=unnormalized_probabilities,
                                  pop_values=pop_values)
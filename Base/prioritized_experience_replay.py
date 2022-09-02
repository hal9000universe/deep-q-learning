# py
from typing import Tuple

# nn & rl
import numpy as np
from numpy import power, take_along_axis, zeros, float64
from numpy.random import uniform

# lib
from replay_buffer import ReplayBuffer
from sum_tree import *


class PrioritizedExperienceReplay(ReplayBuffer):
    _buffer_size: int
    _tree: ndarray
    _priorities: ndarray
    _states: ndarray
    _actions: ndarray
    _rewards: ndarray
    _observations: ndarray
    _dones: ndarray
    _counter: int
    _num_samples: int
    _alpha: float
    _beta: float
    _min_priority: float

    def __init__(self,
                 buffer_size: int,
                 obs_placeholder_shape: Tuple,
                 ac_placeholder_shape: Tuple,
                 alpha: float,
                 beta: float,
                 min_priority: float,
                 ):
        super(PrioritizedExperienceReplay, self).__init__(buffer_size, obs_placeholder_shape, ac_placeholder_shape)
        self._tree = gen_tree(buffer_size)
        self._priorities = zeros((buffer_size,), dtype=float64)
        self._alpha = alpha
        self._beta = beta
        self._min_priority = min_priority

    def _update(self, index: int, priority: float):
        update(self._tree[self._buffer_size: 2*self._buffer_size - 1][index], self._adjust_priority(priority))

    def add_experience(self,
                       priority: float,
                       state: ndarray,
                       action: int,
                       reward: float,
                       observation: ndarray,
                       done: bool
                       ):
        index: int = self._counter % self._buffer_size
        self._priorities[index] = priority
        self._update(index, priority)
        self.add(state, action, reward, observation, done)

    def _adjust_priority(self, priority: float) -> float:
        return power(priority + self._min_priority, self._alpha)


def sample(num_samples: int,
           tree: ndarray,
           priorities: ndarray,
           states: ndarray,
           actions: ndarray,
           rewards: ndarray,
           observations: ndarray,
           dones: ndarray,
           batch_size: int,
           alpha: float
           ) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    max_val: float = tree[0]
    values: ndarray = uniform(0.0, max_val, batch_size)
    indices: ndarray = retrieve(tree, values)
    priority_samples: ndarray = take_along_axis(priorities, indices)
    state_samples: ndarray = take_along_axis(states, indices)
    action_samples: ndarray = take_along_axis(actions, indices)
    reward_samples: ndarray = take_along_axis(rewards, indices)
    observation_samples: ndarray = take_along_axis(observations, indices)
    done_samples: ndarray = take_along_axis(dones, indices)
    importance_sampling_weights: ndarray = power(num_samples * priority_samples, alpha) / np.max(priority_samples)
    return state_samples, action_samples, reward_samples, observation_samples, done_samples, importance_sampling_weights

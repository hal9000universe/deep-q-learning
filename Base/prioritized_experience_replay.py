# py
from typing import Tuple

# nn & rl
import numpy as np
from numpy import power, take_along_axis, zeros, float32, reshape

# lib
from Base.replay_buffer import ReplayBuffer
from Base.sum_tree import *


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
                 obs_shape: Tuple,
                 ac_shape: Tuple,
                 alpha: float,
                 beta: float,
                 min_priority: float = 0.01,
                 ):
        super(PrioritizedExperienceReplay, self).__init__(buffer_size, obs_shape, ac_shape)
        self._tree = gen_tree(buffer_size)
        self._priorities = zeros((buffer_size,), dtype=float32)
        self._alpha = alpha
        self._beta = beta
        self._min_priority = min_priority

    @property
    def tree(self) -> ndarray:
        return self._tree

    @property
    def priorities(self) -> ndarray:
        return self._priorities

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self):
        return self.beta

    def update(self, index: int, priority: float):
        update(self._tree, index, self._adjust_priority(priority))

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
        self.update(index, priority)
        self.add(state, action, reward, observation, done)

    def _adjust_priority(self, priority: float) -> float:
        return power(priority + self._min_priority, self._alpha)


@numba.njit
def sample_batch(num_samples: int,
                 priorities: ndarray,
                 states: ndarray,
                 actions: ndarray,
                 rewards: ndarray,
                 observations: ndarray,
                 dones: ndarray,
                 indices: ndarray,
                 batch_size: int,
                 alpha: float
                 ) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    expanded_indices: ndarray = reshape(indices, (batch_size, 1))
    priority_samples: ndarray = take_along_axis(priorities, indices, 0)
    state_samples: ndarray = take_along_axis(states, expanded_indices, 0)
    action_samples: ndarray = take_along_axis(actions, indices, 0)
    reward_samples: ndarray = take_along_axis(rewards, indices, 0)
    observation_samples: ndarray = take_along_axis(observations, expanded_indices, 0)
    done_samples: ndarray = take_along_axis(dones, indices, 0)
    importance_sampling_weights: ndarray = power(num_samples * priority_samples, alpha) / np.max(priority_samples)
    return state_samples, action_samples, reward_samples, observation_samples, done_samples, importance_sampling_weights

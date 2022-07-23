from typing import Tuple
from numpy import ndarray, zeros, ones, float64, int64, random


class ReplayBuffer:
    _buffer_size: int
    _states: ndarray[float64]
    _actions: ndarray[int64]
    _rewards: ndarray[float64]
    _observations: ndarray[float64]
    _dones: ndarray[bool]
    _counter: int
    _samples: int

    def __init__(self,
                 obs_shape: Tuple[int],
                 ac_shape: Tuple[int],
                 buffer_size: int = 1000000):
        self._buffer_size = buffer_size
        self._states = zeros(obs_shape, dtype=float64)
        self._actions = ones(ac_shape, dtype=int64)
        self._rewards = zeros((buffer_size,), dtype=float64)
        self._observations = zeros(obs_shape, dtype=float64)
        self._dones = zeros((buffer_size,), dtype=bool)
        self._counter = 0
        self._samples = 0

    @property
    def size(self) -> int:
        return self._samples

    def add(self, state: ndarray, action: int, reward: float, observation: ndarray, done: bool):
        self._states[self._counter % self._buffer_size] = state
        self._actions[self._counter % self._buffer_size] = action
        self._rewards[self._counter % self._buffer_size] = reward
        self._observations[self._counter % self._buffer_size] = observation
        self._dones[self._counter % self._buffer_size] = done
        self._counter += 1
        self._samples = min(self._counter, self._buffer_size)

    def sample_batch(self, batch_size: int = 64) -> Tuple[ndarray[float64], ndarray[int64], ndarray[float64],
                                                          ndarray[float64], ndarray[bool]]:
        random_indices: ndarray[int] = random.randint(0, self._samples - 1, batch_size)
        batch = (self._states[random_indices], self._actions[random_indices], self._rewards[random_indices],
                 self._observations[random_indices], self._dones[random_indices])
        return batch

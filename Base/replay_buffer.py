# py
import numba
from typing import Tuple

# nn & rl
from numpy import zeros, ndarray, int64, float64
from numpy.random import randint


class ReplayBuffer:
    _buffer_size: int
    _states: ndarray
    _actions: ndarray
    _rewards: ndarray
    _observations: ndarray
    _dones: ndarray
    _counter: int
    _num_samples: int

    def __init__(self,
                 buffer_size: int,
                 obs_placeholder_shape: Tuple,
                 ac_placeholder_shape: Tuple
                 ):
        self._buffer_size = buffer_size
        self._states = zeros(obs_placeholder_shape, dtype=float64)
        self._actions = zeros(ac_placeholder_shape, dtype=int64)
        self._rewards = zeros((buffer_size,), dtype=float64)
        self._observations = zeros(obs_placeholder_shape, dtype=float64)
        self._dones = zeros((buffer_size,), dtype=bool)
        self._counter = 0
        self._num_samples = 0

    @property
    def size(self) -> int:
        return self._num_samples

    @property
    def states(self) -> ndarray:
        return self._states

    @property
    def actions(self) -> ndarray:
        return self._actions

    @property
    def rewards(self) -> ndarray:
        return self._rewards

    @property
    def observations(self) -> ndarray:
        return self._observations

    @property
    def dones(self) -> ndarray:
        return self._dones

    def add(self, state: ndarray, action: int, reward: float, observation: ndarray, done: bool):
        self._states[self._counter % self._buffer_size] = state
        self._actions[self._counter % self._buffer_size] = action
        self._rewards[self._counter % self._buffer_size] = reward
        self._observations[self._counter % self._buffer_size] = observation
        self._dones[self._counter % self._buffer_size] = done
        self._counter += 1
        self._num_samples = min(self._counter, self._buffer_size)


@numba.njit
def sample_batch(num_samples: int,
                 states: ndarray,
                 actions: ndarray,
                 rewards: ndarray,
                 observations: ndarray,
                 dones: ndarray,
                 batch_size: int
                 ) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    random_indices: ndarray = randint(0, num_samples, batch_size)
    batch = (
        states[random_indices],
        actions[random_indices],
        rewards[random_indices],
        observations[random_indices],
        dones[random_indices]
    )
    return batch

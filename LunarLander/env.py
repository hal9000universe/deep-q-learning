# py
from typing import Any, Tuple

# nn & rl
import gym
from numpy import ndarray, append, newaxis


class ObsWrapper(gym.Wrapper):
    _step: int
    _max_steps: int

    def __init__(self, environment: gym.Env, max_steps: int):
        super(ObsWrapper, self).__init__(environment)
        self._step = 0
        self._max_steps = max_steps
        self._observation_space = gym.spaces.Box(shape=(1, 9), low=float('-inf'), high=float('inf'))

    def observation(self, observation: ndarray) -> ndarray:
        fraction_finished: float = self._step / self._max_steps
        return append(observation, fraction_finished)[newaxis, ...]

    def step(self, action) -> Tuple[ndarray, float, bool, Any]:
        self._step += 1
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def reset(self, **kwargs) -> ndarray:
        self._step = 0
        state: ndarray = self.env.reset()
        return self.observation(state)

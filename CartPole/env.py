# py
from typing import Any, Tuple

# nn & rl
import gym
from numpy import ndarray, newaxis


class ObsWrapper(gym.Wrapper):

    def __init__(self, environment: gym.Env):
        super(ObsWrapper, self).__init__(environment)
        self._observation_space = gym.spaces.Box(shape=(1, 4), low=float('-inf'), high=float('inf'))

    @staticmethod
    def observation(observation: ndarray) -> ndarray:
        return observation[newaxis, ...]

    def step(self, action) -> Tuple[ndarray, float, bool, Any]:
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def reset(self, **kwargs) -> ndarray:
        state: ndarray = self.env.reset()
        return self.observation(state)

# nn & rl
import gym
from numpy import ndarray, newaxis, float64


def transform(obs: ndarray) -> ndarray:
    obs = obs.__array__(float64)
    obs: ndarray = 0.3 * obs[:, :, :, 0] + 0.6 * obs[:, :, :, 1] + 0.1 * obs[:, :, :, 2]
    obs = obs[newaxis, ...]
    return obs


def create_env() -> gym.Env:
    env: gym.Env = gym.make('ALE/Breakout-v5')
    env = gym.wrappers.FrameStack(env, 4)
    env = gym.wrappers.TransformObservation(env, transform)
    return env

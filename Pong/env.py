# nn & rl
import gym
from numpy import ndarray, newaxis, float32, reshape
from cv2 import resize


def greyscale(obs: ndarray) -> ndarray:
    obs: ndarray = 0.3 * obs[:, :, 0] + 0.6 * obs[:, :, 1] + 0.1 * obs[:, :, 2]
    return obs


def rescale(obs: ndarray) -> ndarray:
    obs = resize(obs, (84, 84))
    return obs


def chunk(obs: ndarray) -> ndarray:
    obs: ndarray = obs.__array__(float32)
    (frames, H, W) = obs.shape
    P: int = 14
    N: int = int(H * W / (P**2))
    obs = reshape(obs, (P, P, N * frames))
    obs = obs[newaxis]
    return obs


def create_env() -> gym.Env:
    env: gym.Env = gym.make('ALE/Pong-v5',
                            full_action_space=False)
    env = gym.wrappers.TransformObservation(env, rescale)
    env = gym.wrappers.TransformObservation(env, greyscale)
    env = gym.wrappers.FrameStack(env, 4)
    env = gym.wrappers.TransformObservation(env, chunk)
    return env

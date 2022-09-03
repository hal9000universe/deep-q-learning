# py
from typing import Callable

# nn & rl
import gym
import haiku as hk
from numpy import ndarray, argmax


def generate_visualization(environment: gym.Env, network: hk.Transformed) -> Callable:
    def visualize_agent(params: hk.Params):
        state: ndarray = environment.reset()
        done: bool = False
        while not done:
            action: int = int(argmax(network.apply(params, state)))
            state, reward, done, info = environment.step(action)
            environment.render()

    return visualize_agent

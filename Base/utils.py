# py
import os
from typing import Callable
from time import time
from pickle import dump, load

# nn & rl
import gym
import haiku as hk
from numpy import ndarray, argmax


def stop_time(name: str, fun: Callable, *args):
    start: float = time()
    fun(*args)
    stop: float = time()
    print("{}: {}s".format(name, stop - start))


def generate_saving(directory: str) -> Callable:
    async def save_state(params: hk.Params):
        if not os.path.exists(directory):
            os.mkdir(directory)
        with open(os.path.join(directory, "params.pickle"), "wb") as file:
            dump(params, file)

    return save_state


def generate_loading(directory: str) -> Callable:
    def load_state() -> hk.Params:
        with open(os.path.join(directory, "params.pickle"), "rb") as file:
            params: hk.Params = load(file)
        return params

    return load_state


def generate_visualization(environment: gym.Env, network: hk.Transformed) -> Callable:
    def visualize_agent(params: hk.Params):
        state: ndarray = environment.reset()
        done: bool = False
        while not done:
            action: int = int(argmax(network.apply(params, state)))
            state, reward, done, info = environment.step(action)
            environment.render()

    return visualize_agent

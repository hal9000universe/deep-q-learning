# py
import os
from typing import Callable, Any, Mapping, Tuple
from time import time
from pickle import dump, load

# nn & rl
import gym
import haiku as hk
from numpy import ndarray, argmax


def stop_time(name: str, fun: Callable, *args) -> Any:
    start: float = time()
    out = fun(*args)
    stop: float = time()
    print("{}: {}s".format(name, stop - start))
    return out


def generate_saving(directory: str) -> Callable:
    async def save_state(params: hk.Params, opt_state: Mapping):
        if not os.path.exists(directory):
            os.mkdir(directory)
        with open(os.path.join(directory, "params.pickle"), "wb") as file:
            dump(params, file)
        with open(os.path.join(directory, "opt_state.pickle"), "wb") as file:
            dump(opt_state, file)
    return save_state


def generate_loading(directory: str) -> Callable:
    def load_state() -> Tuple[hk.Params, Mapping]:
        with open(os.path.join(directory, "params.pickle"), "rb") as file:
            params: hk.Params = load(file)
        with open(os.path.join(directory, "opt_state.pickle"), "rb") as file:
            opt_state: Mapping = load(file)
        return params, opt_state

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

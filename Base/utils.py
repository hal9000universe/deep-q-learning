# py
import os
from typing import Callable
from time import time
from pickle import dump, load

# nn & rl
import haiku as hk


def stop_time(name: str, fun: Callable, *args):
    start: float = time()
    fun(*args)
    stop: float = time()
    print("{}: {}s".format(name, stop - start))


async def save_state(params: hk.Params):
    if not os.path.exists("lunar_lander"):
        os.mkdir("lunar_lander")
    with open("lunar_lander/params.pickle", "wb") as file:
        dump(params, file)


def load_state() -> hk.Params:
    with open("lunar_lander/params.pickle", "rb") as file:
        params: hk.Params = load(file)
    return params

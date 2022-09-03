# py
from typing import Callable
from time import time


def stop_time(name: str, fun: Callable, *args):
    start: float = time()
    fun(*args)
    stop: float = time()
    print("{}: {}s".format(name, stop - start))

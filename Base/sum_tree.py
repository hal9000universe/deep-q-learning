# py
import math
import numba

# nn & rl
import numpy
from numpy import ndarray


@numba.njit
def propagate_changes(tree: ndarray, node: int, change: float):
    size = int(tree.size / 2)
    node += size
    while node >= 1:
        tree[node] += change
        node = math.floor(node / 2)


@numba.njit
def update(tree: ndarray, node: int, new_value: float):
    node += 1
    size = int(tree.size / 2)
    change = new_value - tree[node + size]
    propagate_changes(tree, node, change)


@numba.njit
def retrieve(tree: ndarray, value):
    i = 1
    size = int(tree.size / 2)
    while i + 1 <= size:
        if tree[2 * i] >= value or tree[2 * i + 1] == 0:
            i = 2 * i
        else:
            value -= tree[2 * i]
            i = 2 * i + 1
    return i - size - 1


def gen_tree(buffer_size: int) -> ndarray:
    return numpy.zeros((2 * buffer_size,))

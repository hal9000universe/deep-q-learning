import numpy
from numpy import ndarray
import math


def propagate_changes(tree: ndarray, node: int, change: float):
    size = tree.size / 2
    node += size - 1
    while node > 0:
        tree[node] += change
        node = math.floor(node / 2)


def add(tree: ndarray, node: int, new_value: float):
    size = tree.size / 2
    change = new_value - tree[node + size - 1]
    tree[node + size - 1] = new_value
    propagate_changes(tree, node, change)


def retrieve(tree: ndarray, value: float) -> int:
    i = 0
    size = tree.size / 2
    while 2*i+2 <= size:
        if tree[2*i+1] >= value or tree[2*i+2] == 0:
            i = 2*i+1
        else:
            value -= tree[2*i+1]
            i = 2*i+2

    return i - size + 1


def gen_tree(buffer_size: int) -> ndarray:
    return numpy.zeros((2*buffer_size,))









# py
from typing import Tuple, List, Dict, Callable

# nn & rl
import jax
import optax
import haiku as hk
import jax.numpy as jnp
from numpy import ndarray, average, array, std
from numpy.linalg import norm


def generate_forward_analysis(network: hk.Transformed) -> Callable:
    @jax.jit
    def forward_analysis(params: hk.Params, inp: ndarray) -> Tuple[ndarray, ndarray]:
        pred, features = network.apply(params, inp, True)
        return pred, features

    return forward_analysis


@jax.jit
def loss_metric(y_pred: ndarray, y_true: ndarray) -> ndarray:
    loss: ndarray = jnp.mean(optax.huber_loss(y_pred, y_true))
    return loss


def data_means(data: Dict[int, List[ndarray]]) -> ndarray:
    class_means: List[ndarray] = list()
    for value in data.values():
        avg: ndarray = average(value, axis=0)
        class_means.append(avg)

    return array(class_means)


def mean_eq_dist(means: ndarray, axis: int) -> Tuple[bool, ndarray]:
    eps: float = 1e-03
    total_mean = average(means, axis=axis)
    class_means = means - total_mean
    lengths: List[float] or ndarray = []
    for cls in class_means:
        length: float = norm(cls)
        lengths.append(length)
    lengths = array(lengths)
    stand_dev: ndarray = std(lengths, axis=0)
    return stand_dev < eps, stand_dev

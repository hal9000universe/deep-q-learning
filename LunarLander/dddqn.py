# py
from typing import Tuple

# nn & rl
import jax
import haiku as hk
import jax.numpy as jnp
from numpy import ndarray


class Model(hk.Module):
    _lin1: hk.Linear
    _lin2: hk.Linear
    _val: hk.Linear
    _adv: hk.Linear

    def __init__(self, num_actions: int):
        super().__init__()
        self._lin1 = hk.Linear(32)
        self._lin2 = hk.Linear(64)
        self._val = hk.Linear(1)
        self._adv = hk.Linear(num_actions)

    def __call__(self, x: ndarray, return_features: bool = False) -> ndarray or Tuple[ndarray, ndarray]:
        x = self._lin1(x)
        x = jax.nn.relu(x)
        x = self._lin2(x)
        x = jax.nn.relu(x)
        val: ndarray = self._val(x)
        adv: ndarray = self._adv(x)
        Q: ndarray = val + adv - jnp.mean(adv, axis=1, keepdims=True)
        if return_features:
            return Q, x
        return Q

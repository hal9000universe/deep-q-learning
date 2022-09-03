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
        self._lin1 = hk.Linear(64)
        self._lin2 = hk.Linear(64)
        self._val = hk.Linear(1)
        self._adv = hk.Linear(num_actions)

    def __call__(self, x: ndarray or jnp.ndarray) -> ndarray or jnp.ndarray:
        x = self._lin1(x)
        x = jax.nn.relu(x)
        x = self._lin2(x)
        x = jax.nn.relu(x)
        val: ndarray or jnp.ndarray = self._val(x)
        adv: ndarray or jnp.ndarray = self._adv(x)
        Q: ndarray or jnp.ndarray = val + adv - jnp.mean(adv, axis=1, keepdims=True)
        return Q

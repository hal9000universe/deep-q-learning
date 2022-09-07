# nn & rl
import jax
import haiku as hk
import jax.numpy as jnp
from numpy import ndarray


class Model(hk.Module):
    _lin1: hk.Linear
    _lin2: hk.Linear
    _lin3: hk.Linear
    _lin4: hk.Linear

    def __init__(self, num_actions: int):
        super().__init__()
        self._lin1 = hk.Linear(64)
        self._lin2 = hk.Linear(64)
        self._lin3 = hk.Linear(64)
        self._lin4 = hk.Linear(num_actions)

    def __call__(self, x: ndarray or jnp.ndarray) -> ndarray or jnp.ndarray:
        x = self._lin1(x)
        x = jax.nn.relu(x)
        x = self._lin2(x)
        x = jax.nn.relu(x)
        x = self._lin3(x)
        x = jax.nn.relu(x)
        x = self._lin4(x)
        return x

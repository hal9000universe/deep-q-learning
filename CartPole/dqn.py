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
        self._lin1 = hk.Linear(64, w_init=hk.initializers.RandomUniform(-0.03, 0.03))
        self._lin2 = hk.Linear(num_actions, w_init=hk.initializers.RandomUniform(-0.03, 0.03))

    def __call__(self, x: ndarray or jnp.ndarray) -> ndarray or jnp.ndarray:
        x = self._lin1(x)
        x = jax.nn.relu(x)
        x = self._lin2(x)
        return x

import haiku as hk
import jax
import time
import numpy as np
from typing import Tuple


class FeedForward(hk.Module):

    def __init__(self, n1: int, n2: int):
        super().__init__()
        self._lin1 = hk.Linear(n1)
        self._lin2 = hk.Linear(n2)

    def __call__(self, x):
        x = self._lin1(x)
        x = jax.nn.relu(x)
        x = self._lin2(x)
        return x


class TransformerLayer(hk.Module):

    def __init__(self, mha1: Tuple, ffn1: Tuple):
        super().__init__()
        self._mha1 = hk.MultiHeadAttention(*mha1)
        self._norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self._norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self._ffn1 = FeedForward(*ffn1)

    def __call__(self, inp):
        attn_output = self._mha1(inp, inp, inp)
        out1 = self._norm1(inp + attn_output)
        ffn_out = self._ffn1(out1)
        out2 = self._norm2(out1 + ffn_out)  # TODO: add dropout
        return out2


if __name__ == '__main__':
    num_heads: int = 2
    key_size: int = 2
    w_init_scale: float = 1.0
    mha: Tuple = (num_heads, key_size, w_init_scale)
    dff: int = 256
    d_model: int = 4
    n: Tuple = (dff, d_model)

    transformer_layer = hk.without_apply_rng(hk.transform(lambda *args: TransformerLayer(
        mha, n
    )(*args)))

    rng: jax.random.PRNGKeyArray = jax.random.PRNGKey(time.time_ns())
    test_inp = np.zeros((84, 84, 4), dtype=np.float32)

    params = transformer_layer.init(rng, test_inp)

    output = transformer_layer.apply(params, test_inp)
    print(output.shape)

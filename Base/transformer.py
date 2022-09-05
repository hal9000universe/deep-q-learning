# py
import time
from typing import Tuple

# nn & rl
import jax
import haiku as hk
from numpy import ndarray, zeros, float32


class FeedForward(hk.Module):
    _lin1: hk.Linear
    _lin2: hk.Linear

    def __init__(self, n1: int, n2: int):
        super().__init__()
        self._lin1 = hk.Linear(n1)
        self._lin2 = hk.Linear(n2)

    def __call__(self, x: ndarray) -> ndarray:
        x = self._lin1(x)
        x = jax.nn.relu(x)
        x = self._lin2(x)
        return x


class TransformerLayer(hk.Module):
    _mha: hk.MultiHeadAttention
    _norm1: hk.LayerNorm
    _norm2: hk.LayerNorm
    _ffn: FeedForward

    def __init__(self, mha: Tuple, ffn: Tuple, rate: float):
        super().__init__()
        self._mha = hk.MultiHeadAttention(*mha)
        self._norm1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self._norm2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self._ffn = FeedForward(*ffn)
        self._rate = rate

    def __call__(self, inp: ndarray) -> ndarray:
        attn_output = self._mha(inp, inp, inp)
        out1 = self._norm1(inp + attn_output)
        ffn_out = self._ffn(out1)
        rng_key = jax.random.PRNGKey(time.time_ns())
        ffn_out = hk.dropout(rng_key, self._rate, ffn_out)
        out2 = self._norm2(out1 + ffn_out)
        return out2


if __name__ == '__main__':
    num_heads: int = 2
    key_size: int = 2
    w_init_scale: float = 1.0
    mha_in: Tuple = (num_heads, key_size, w_init_scale)
    dff: int = 256
    d_model: int = 4
    n: Tuple = (dff, d_model)
    dropout_rate: float = 0.2

    transformer_layer: hk.Transformed = hk.without_apply_rng(
        hk.transform(
            lambda *args: TransformerLayer(mha_in, n, dropout_rate)(*args)
        )
    )

    rng: jax.random.PRNGKeyArray = jax.random.PRNGKey(time.time_ns())
    test_inp: ndarray = zeros((84, 84, 4), dtype=float32)

    params: hk.Params = transformer_layer.init(rng, test_inp)

    output: ndarray = transformer_layer.apply(params, test_inp)
    print(output.shape)

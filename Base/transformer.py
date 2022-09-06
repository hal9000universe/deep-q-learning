# py
import time
from typing import Tuple, List

# nn & rl
import jax
import haiku as hk
from numpy import ndarray


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


class Transformer(hk.Module):
    _layers: List
    _flatten: hk.Flatten
    _lin: hk.Linear

    def __init__(self,
                 num_layers: int,
                 init_params: List,
                 num_actions: int
                 ):
        super(Transformer, self).__init__()
        self._layers = []
        for layer in range(num_layers):
            transformer_layer: TransformerLayer = TransformerLayer(*init_params[layer])
            self._layers.append(transformer_layer)
        self._flatten = hk.Flatten()
        self._lin = hk.Linear(num_actions)

    def __call__(self, inp: ndarray) -> ndarray:
        for layer in self._layers:
            inp = layer(inp)
        out = self._flatten(inp[:, :, :, 0])
        out = self._lin(out)
        return out

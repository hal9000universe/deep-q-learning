# py
from typing import List, Tuple

# lib
from Base.transformer import Transformer


class PongFormer(Transformer):

    def __init__(self):
        num_layers: int = 2
        num_actions: int = 6
        num_heads: int = 12
        key_size: int = 12
        w_init_scale: float = 1.0
        mha: Tuple = (num_heads, key_size, w_init_scale)
        dff: int = 256
        d_model: int = 144
        num_neurons: Tuple = (dff, d_model)
        dropout_rate: float = 0.2
        init_params: List = [(mha, num_neurons, dropout_rate) for _ in range(num_layers)]
        super(PongFormer, self).__init__(
            num_layers=num_layers,
            init_params=init_params,
            num_actions=num_actions,
        )

import jax
import jax.random as jr
# import equinox as eqx
import equinox.nn as nn
from itertools import chain

from typing import Callable, List, Optional


class MLP(nn.Sequential):
    def __init__(
        self,
        input_size: int,
        features: List[int],
        emb_size: Optional[int] = None,
        activation_fn: Callable = jax.nn.relu,
        *,
        key: jr.PRNGKeyArray,
        # **kwargs,
    ):
        key_list = jr.split(key, len(features))
        in_sizes = list(chain([input_size], features[:-1]))

        layers = []
        for i in range(len(features)):
            layers.append(nn.Linear(in_sizes[i], features[i], use_bias=True, key=key_list[i]))
            layers.append(nn.Lambda(activation_fn))

        layers.append(nn.Linear(features[-1], emb_size, use_bias=False))

        super().__init__(layers)

from abc import ABC, abstractmethod
# from typing import Callable, Tuple

import equinox as eqx
import jax.tree_util as jtu
import jax.random as jr
from jaxtyping import PyTree
from jax.experimental import io_callback


class DataModule(ABC, eqx.Module):
    @abstractmethod
    def init(self, stage: str, key: jr.KeyArray):
        raise NotImplementedError

    @abstractmethod
    def next(self, state: PyTree):
        raise NotImplementedError


class callback_loader:
    """
    Wrap a dataloader in an io_callback.

    This allows us to use it inside jit-compiled scan loops. We use a class because pickle does not
    work on lambdas and it is needed when replicating dataloaders across multiple workers.

    TODO: Currently this doens't work with the progress bar implemented in 'src.train.utils'.
    """
    def __init__(self, loader):
        iterator = iter(loader)
        example_batch = next(iterator)
        self.iter_data = jtu.Partial(io_callback, lambda: next(iterator), example_batch)

    def __call__(self, *_):
        return self.iter_data()


class identity_collate:
    def __call__(self, x):
        return x

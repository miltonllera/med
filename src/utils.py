from abc import ABC, abstractmethod
from typing import Any

import jax.tree_util as jtu
from jaxtyping import Array, Float


TENSOR = Float[Array, "..."]


class jitted_method:
    def __init__(self, method):
        self.method = method
        if getattr(self.method, "__isabstractmethod__", False):
            self.__isabstractmethod__ = self.method.__isabstractmethod__

    def __get__(self, instance, owner):
        if instance is None:
            return self.method
        return jtu.Partial(self.method, instance)


class StateIndexer(ABC):
    @abstractmethod
    def __call__(self, trainer_state) -> Any:
        raise NotImplementedError

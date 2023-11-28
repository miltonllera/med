from abc import ABC, abstractmethod
from functools import partial, wraps
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float
from tqdm import tqdm


TENSOR = Float[Array, "..."]


def tree_stack(trees):
    return jtu.tree_map(lambda *v: jnp.stack(v), *trees)


def tree_unstack(tree):
    leaves, treedef = jtu.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


def loop(step_fn, state, n_iters):
    accumulated_results = []

    for i in tqdm(range(n_iters)):
        state, results = step_fn(state, i)
        accumulated_results.append(results)

    stacked_results = tree_stack(accumulated_results)

    return state, stacked_results


def jit_method(*args, **kwargs):

    if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], Callable):
        return partial(jax.jit, static_argnames=("self",))(args[0])

    @wraps
    def wrapper(f):
        return partial(jax.jit, *args, **kwargs)(f)

    return wrapper


class jax_partial:
    """
    Use this class if you want to mark a method as a valid input for a jax transform.

    Standard partial functions cannot be used as arguments to jitted functions. For this, we must
    make use of 'jax.tree_util.Partial', which makes the function a traceable value. This method
    overrides the get method of a class attribute to wrap it in a partial at runtime.
    """
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

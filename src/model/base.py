from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from typing import Any, Callable, Iterable, NamedTuple, Tuple
from typing_extensions import Self
from jaxtyping import Array, Float, Int, PyTree


class State(NamedTuple):
    node_states: PyTree
    input_embedding: Float[Array, "..."]
    dev_steps: Int
    rng_key: jr.KeyArray


class FunctionalModel(ABC):
    @abstractmethod
    def init(self, inputs, key) -> PyTree:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, inputs: PyTree, key: jr.KeyArray) -> Any:
        raise NotImplementedError

    @abstractmethod
    def partition(self) -> PyTree:
        """Define how the model should partitioned between params and statics"""
        raise NotImplementedError

    @abstractmethod
    def parameters(self) -> PyTree:
        raise NotImplementedError

    @abstractmethod
    def set_parameters(self, params) -> Self:
        raise NotImplementedError

    @abstractmethod
    def set_inference(self, mode=True) -> Self:
        raise NotImplementedError

    def set_train(self) -> Self:
        return self.set_inference(False)


class DevelopmentalModel(eqx.Module, FunctionalModel):
    """
    A generic developmental model which takes an input encoding (aka goal, "DNA", etc.) and
    produces an output by growing it over several steps.
    """
    input_encoder: Callable
    output_decoder: Callable
    dev_steps: Int
    inference: bool
    output_dev_states: bool

    @jax.named_scope("src.model.DevoModel.rollout")
    def rollout(
        self,
        inputs: Float[Array, "..."],
        key: jr.KeyArray
    )-> Tuple[Float[Array, "..."], Iterable[State]]:
        if isinstance(self.dev_steps, (tuple, list)):
            max_dev_steps = self.dev_steps[1]
        else:
            max_dev_steps = self.dev_steps

        # c_key, init_key = jr.split(key, 2)

        # Set the input embedding and initial rng key
        init_state = self.init(inputs, key)

        # TODO: Sample the number of generation steps
        final_state, states = jax.lax.scan(self.step, init_state, jnp.arange(max_dev_steps))

        # TODO: Do we want to have some decoding model here? For example to project from hidden
        # states to output space? Otherwise the dimensionality for images might be too high.
        output = self.output_decoder(final_state.node_states)

        return output, states

    @abstractmethod
    def step(self, carry: State, _, *args) -> Tuple[State, Iterable[State]]:
        raise NotImplementedError

    def partition(self):
        """Define how the model should partitioned between params and statics"""
        return eqx.partition(self, eqx.is_array)

    def parameters(self) -> PyTree:
        return self.partition()[0]

    def set_parameters(self, params) -> Self:
        return eqx.combine(params, self)

    def set_inference(self, mode=True) -> Self:
        model = eqx.tree_at(lambda x: x.inference, self, mode)
        return eqx.tree_inference(model, mode)

    def set_train(self) -> Self:
        return self.set_inference(False)

    def return_dev_states(self, mode: bool) -> Self:
        return eqx.tree_at(lambda x: x.output_dev_states, self, mode)

    __call__ = rollout

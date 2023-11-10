from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Int

from src.utils import TENSOR
from src.model.base import DevelopmentalModel, State


class NCA(DevelopmentalModel):
    """
    Neural Cellular Automata based on Mordvintsev et al. (2020) which supports using goal-directed
    generation as in Shudhakaran et al. (2022).

    This class assumes a grid like organization where cell states occupy the leading dimension of
    the vectors. This means that we can use convolution operations for the updates themselves and
    any function that rearranges the dimensions internally must reverse this process when returning
    the results back to the NCA class.
    """
    # functions
    state_size: int
    grid_size: Tuple[int, int]
    alive_fn: Callable
    context_fn: Callable
    message_fn: Callable
    update_fn: Callable
    # hyperparams
    update_prob: float

    def __init__(
        self,
        state_size: int,
        grid_size: Tuple[int, int],
        input_encoder: Callable,
        output_decoder: Callable,
        alive_fn: Callable,
        context_fn: Callable,
        message_fn: Callable,
        update_fn: Callable,
        update_prob: float,
        dev_steps: int,
        output_dev_states: bool = False
    ):
        super().__init__(input_encoder, output_decoder, dev_steps, False, output_dev_states)
        self.state_size = state_size
        self.grid_size = grid_size
        self.alive_fn = alive_fn
        self.context_fn = context_fn
        self.message_fn = message_fn
        self.update_fn = update_fn
        self.update_prob = update_prob

    def init(self, inputs, key):
        H, W = self.grid_size

        key, init_key = jr.split(key)

        # TODO: Random initialization doesn't seem to work. Cells tend to die or their values diverge.
        # init_states = jnp.zeros((self.state_size, H, W))
        # # random initialization of cell
        # seed = (0.5 * jr.normal(init_key, (self.state_size,))).at[3].set(1)
        # init_states = init_states.at[:, H//2, W//2].set(seed)

        init_states = jnp.zeros((self.state_size, H, W)).at[:, H//2, W//2].set(1.0)
        # jax.debug.print("{}", init_states.max())

        return State(
            input_embedding=self.input_encoder(inputs),
            node_states=init_states,
            rng_key=key,
            dev_steps=self.sample_generation_steps(init_key)
        )

    @jax.named_scope("src.model.NCA.step")
    def step(self, state: Tuple[TENSOR, TENSOR, Int, jr.KeyArray], i: int):
        cell_states, input_embedding, dev_steps, key = state

        def _step():
            update_key, context_key, carry_key = jr.split(key, 3)

            pre_alive_mask = self.alive_fn(cell_states)
            control_signal = self.context_fn(cell_states, input_embedding, key=context_key)

            message_vectors = self.message_fn(
                cell_states + control_signal * pre_alive_mask.astype(jnp.float32)
            )
            updates = self.update_fn(message_vectors)
            new_states = cell_states + updates * self.stochastic_update_mask(update_key)

            alive_mask = (self.alive_fn(new_states) & pre_alive_mask).astype(jnp.float32)
            new_states = new_states * alive_mask

            cell_state = new_states if self.output_dev_states else None  # Only for model analysis

            return State(new_states, input_embedding, dev_steps, carry_key), cell_state

        # NOTE: in this case 'jax.cond' exectutes both branches during evaluation since the
        # functions are not dependent on the input. We could make it short-circuit by passing i
        # to both  branches, however because this code is usually in a vmap, it wouldn't make a
        # difference as it will be translated into a 'jax.select' operation (which executes both
        # branches regardless of value of the condition).
        # see: https://github.com/google/jax/issues/3103, #issuecomment-1716019765
        return jax.lax.cond(
            i < dev_steps,
            _step,
            lambda: (state, cell_states if self.output_dev_states else None)
        )

    def stochastic_update_mask(self, key: jr.PRNGKeyArray):
        return jr.bernoulli(key, self.update_prob, self.grid_size)[jnp.newaxis].astype(jnp.float32)

    def sample_generation_steps(self, key: jr.PRNGKeyArray):
        if isinstance(self.dev_steps, (tuple, list)):
            steps = jax.lax.cond(
                self.inference,
                lambda: self.dev_steps[1],
                lambda: jr.choice(key, jnp.arange(*self.dev_steps)).squeeze(),
            )
        else:
            steps = self.dev_steps
        return steps

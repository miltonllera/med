from src.model.base import State

import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx
import evosax as ex

from jaxtyping import Float, Array, PyTree
from typing import Callable, Optional


def MetaEvolutionTask(
    statics: Callable,
    goals: Float[Array, "n_goals ..."],
    loss_fn: Callable[[State, State], float],
    state_initializer: Callable[[Float[Array, "..."], jr.PRNGKeyArray], State],
    strategy: ex.Strategy,
    strategy_params: Optional[ex.EvoParams] = None,
    inner_generations: int = 200,
    shaper: Callable = lambda x: x
) -> Callable:

    """MetaEvolutionTask. Evaluate parameters of a DevoModel by searching dna sequences
    minimizing loss between final state of development and goal states. Fitness of params
    is the minimum loss reached.
    """

    if strategy_params is None:
        strategy_params = strategy.default_params

    #-------------------------------------------------------------------

    def _inner_loop(key, params, goal):
        """run the inner evolution loop with provided strategy and return minimum loss obtained"""
        model = eqx.combine(params, statics)

        def _inner_step(carry, x):
            """inner evolutionay step"""
            [es_state, key, goal] = carry
            key, ask_key, init_key, model_key = jr.split(key, 4)

            x, es_state = strategy.ask(ask_key, es_state, strategy_params)
            dna = shaper(x)

            init_state = state_initializer(dna, init_key)

            output, _ = jax.vmap(model, in_axes=(0, 0, None))(
                dna, init_state, model_key,
            )
            loss = jax.vmap(loss_fn, in_axes=(0, None))(output, goal)
            es_state = strategy.tell(x, loss, es_state, strategy_params)
            return [es_state, key, goal], loss

        key, key_init = jr.split(key)
        es_state = strategy.initialize(key_init, strategy_params)
        _, losses = jax.lax.scan(_inner_step, [es_state, key, goal], jnp.arange(inner_generations))
        return losses.min()

    #-------------------------------------------------------------------

    def _eval(key, params):
        """Evaluation function
        compute the average loss over the goals"""
        return jax.vmap(_inner_loop, in_axes=(None, None, 0))(key, params, goals).mean()

    #-------------------------------------------------------------------

    return _eval


def MetaGATask(
    statics: PyTree,
    goals: Float[Array, "n_goals ..."],
    loss_fn: Callable[[State, State], float],
    state_initializer: Callable[[Float[Array, "..."], jr.PRNGKeyArray], State],
    num_dims: int,
    inner_generations: int = 200,
    popsize: int = 64,
    ga_params: dict = {},
    devo_steps: int = 50,
    shaper: Callable = lambda x: x,
):
    """Summary

    Args:
        statics (Collection): Description
        goals (jax.Array): Description
        num_dims (int): Description
        inner_generations (int, optional): Description
        popsize (int, optional): Description
        ga_params (dict, optional): Description
        shaper (Callable, optional): Description

    Returns:
        TYPE: Description
    """
    strategy = ex.SimpleGA(popsize=popsize, num_dims=num_dims, **ga_params)
    _ga_params = strategy.default_params
    return MetaEvolutionTask(
        statics,
        goals,
        loss_fn,
        state_initializer,
        strategy,
        _ga_params,
        inner_generations,
        devo_steps,
        shaper
    )

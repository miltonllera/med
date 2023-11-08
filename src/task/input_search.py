from typing import Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import evosax as ex
from jaxtyping import Float, PyTree

from src.task.base import Task, MetricCollection
from src.dataset.base import DataModule
from src.model.base import FunctionalModel
from src.utils import jitted_method


# class InputEvoSearch(Task):
#     goals: TENSOR
#     loss_fn: Callable
#     strategy: ex.Strategy
#     strategy_params: Union[ex.EvoParams, Dict]
#     steps: int
#     """
#     Task that, for a set of targets in a dataset, searches over the input space to find the values
#     that minimize the loss of a particular model over said targets.
#     """
#     def __init__(
#         self,
#         goals: TENSOR,
#         loss_fn: Callable,
#         strategy: ex.Strategy,
#         strategy_params: Optional[Union[ex.EvoParams, Dict]] = None,
#         steps: int = 50,
#     ) -> None:
#         super().__init__()

#         if strategy_params is None:
#             strategy_params = strategy.default_params

#         elif isinstance(strategy_params, Dict):
#             default_params = strategy.default_params
#             default_params = default_params.replace(  # type: ignore [reportGeneralTypeIssues]
#                **strategy_params
#             )
#             strategy_params = default_params

#         self.goals = goals
#         self.loss_fn = loss_fn
#         self.strategy = strategy
#         self.strategy_params = strategy_params
#         self.steps = steps

#     def init(self, stage, key):
#         return self.goals

#     def eval(
#         self,
#         model: PyTree,
#         state: PyTree,
#         key: jr.KeyArray,
#     ) -> Float:
#         """
#         Compute a models ability to fit a particular goal by searching over combinations of inputs.
#         """
#         losses, state = self.predict(model, state, key)
#         return losses.mean(), state

#     def predict(
#         self,
#         model: Callable,
#         goals: PyTree,
#         key: jr.KeyArray,
#     ):
#         """
#         Run a search that finds the best combination of parameters for the
#         """

#         def search(goal, key):
#             key, key_init = jr.split(key)
#             goal_search_state = self.strategy.initialize(key_init, self.strategy_params)

#             def _search_step(carry, x):
#                 goal, search_state, key = carry
#                 key, ask_key, model_key = jr.split(key, 3)

#                 dna, search_state = self.strategy.ask(ask_key, search_state, self.strategy_params)

#                 output, _ = jax.vmap(model, in_axes=(0, None))(dna, model_key)

#                 loss = jax.vmap(self.loss_fn, in_axes=(0, None))(output, goal)
#                 search_state = self.strategy.tell(dna, loss, search_state, self.strategy_params)
#                 return (goal, search_state, key), loss

#             _, losses = jax.lax.scan(
#                 _search_step, (goal, goal_search_state, key), jnp.arange(self.steps)
#             )

#             return losses.min(), goal

#         return jax.vmap(search, in_axes=(0, None))(goals, key)


class InputOptimization(Task):
    datamodule: DataModule
    input_size: int
    loss_fn: Callable
    optim: optax.GradientTransformation
    steps: int
    prepare_batch: Callable

    def __init__(
        self,
        datamodule: DataModule,
        input_size: int,
        loss_fn: Callable,
        optim: optax.GradientTransformation,
        steps: int = 50,
        metrics=None,
        prepare_batch=None,
    ) -> None:
        super().__init__()

        if prepare_batch is None:
            prepare_batch = lambda x: x
        if metrics is None:
            metrics = MetricCollection([loss_fn], ['loss'])

        self.datamodule = datamodule
        self.input_size = input_size
        self.loss_fn = loss_fn
        self.optim = optim
        self.steps = steps
        self.metrics = metrics
        self.prepare_batch = prepare_batch

    def goals(self):
        try:
            return jnp.transpose(jnp.stack(self.datamodule.targets), [0, 3, 1, 2])  # type: ignore
        except AttributeError as e:
            e.add_note("Dataset must have a targets field")
            raise e

    @jitted_method
    def init(self, _, key):
        goals = self.goals()
        inputs = jr.normal(key, (len(goals), self.input_size))
        optax_state = self.optim.init(inputs)
        return (inputs, goals), optax_state

    @jitted_method
    def eval(self, model, state, key):
        (inputs, goals), optim_state = state

        @eqx.filter_jit
        def eval_fn(inputs, key):
            batched_keys = jr.split(key, len(inputs))
            outputs, _ = jax.vmap(model)(inputs, batched_keys)
            return jax.vmap(self.loss_fn)(outputs, goals).mean(axis=0)

        @eqx.filter_jit
        def grad_step(carry, _):
            inputs, opt_state, key = carry
            key, eval_key = jr.split(key)

            loss_value, grads = eqx.filter_value_and_grad(eval_fn)(inputs, eval_key)

            updates, opt_state = self.optim.update(grads, opt_state, inputs)
            inputs = eqx.apply_updates(inputs, updates)

            return (inputs, opt_state, key), loss_value

        (inputs, optim_state, key), _ = jax.lax.stop_gradient(
            jax.lax.scan(grad_step, (inputs, optim_state, key), jnp.arange(self.steps))
        )

        return eval_fn(inputs, key), ((inputs, goals), optim_state)

    @jitted_method
    def validate(
        self,
        model: FunctionalModel,
        state: PyTree,
        key: jr.KeyArray
    ):
        (inputs, goals), _ = state
        batched_keys = jr.split(key, len(inputs))

        pred = jax.vmap(model )(inputs, batched_keys)
        pred, goals = self.prepare_batch(pred, goals)

        metrics = self.metrics.compute(pred, goals)

        return metrics, state

    @jitted_method
    def predict(
        self,
        model: FunctionalModel,
        state: PyTree,
        key: jr.KeyArray,
    ):
        inputs = state[0][0]
        batched_keys = jr.split(key, len(inputs))
        return jax.vmap(model)(inputs, batched_keys)[0], state

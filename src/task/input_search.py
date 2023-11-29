from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jaxtyping import PyTree, ArrayLike
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids

from src.task.base import Task, MetricCollection
from src.dataset.base import DataModule
from src.model.base import FunctionalModel
from src.nn.dna import DNADistribution
from src.evo.qd import MAPElites
from src.problem.base import QDProblem

from src.utils import jax_partial


SCORING_FN = Callable[[Any, jr.KeyArray], Tuple]
QD_ALGORITHM = MAPElites  # Use Union to add more algorithms later


class QDSearchDNA(Task):
    """
    Search over possible DNA sequences that guide the rollout of a developmental model. Note that
    unlike other tasks, the models here must be able to provide a set of DNAs (this can be a fixed
    list, a sampling or any other method).
    """
    problem: QDProblem
    qd_algorithm: QD_ALGORITHM
    n_iters: int
    popsize: int
    centroids_fn: Callable
    n_centroid_samples: int

    def __init__(
        self,
        problem,
        qd_algorithm,
        n_iters,
        popsize,
        n_centroids=1000,
        n_centroid_samples=None,
        score_to_coverage_ratio: float = 1.0,
    ) -> None:
        if n_centroid_samples is None:
            n_centroid_samples = problem.descriptor_length * n_centroids

        self.problem = problem
        self.qd_algorithm = qd_algorithm
        self.n_iters = n_iters
        self.popsize = popsize
        self.n_centroids = n_centroids
        self.n_centroid_samples = n_centroid_samples
        self.score_to_coverage_ratio = score_to_coverage_ratio

    # @jit_method
    def init(self, stage, training_state, key):
        if stage == "train":
            return self.init_centroids(key)
        return training_state  # just return the centroids we used for initalization

    @jax_partial
    def eval(
        self,
        model_and_dna: Tuple[FunctionalModel, DNADistribution],
        centroids: PyTree,
        key
    ):
        (_, metrics), _ = self.predict(model_and_dna, centroids, key)  # type: ignore

        qd_score = metrics['qd_score'][-1]
        coverage = metrics['coverage'][-1]

        total_score = self.score_to_coverage_ratio * qd_score + coverage

        return total_score, centroids

    @jax_partial
    def validate(
        self,
        model_and_dna,
        centroids,
        key
    ):
        (_, metrics), (mpe_state, _) = self.predict(
            model_and_dna, centroids, key
        ) # type: ignore
        qd_score = metrics['qd_score'][-1]
        coverage = metrics['coverage'][-1]
        total_score = self.score_to_coverage_ratio * qd_score + coverage
        metrics['loss'] = total_score

        bd_limits = (self.problem.descriptor_min_val, self.problem.descriptor_max_val)
        repertoire = mpe_state[0]

        extra_results = (repertoire, bd_limits)

        return (metrics, extra_results), centroids

    @jax_partial
    def predict(self, model_and_dna, centroids, key):
        model, dna_distribution = model_and_dna

        @jax.vmap
        def _eval(genotype, key):
            output, _ = model(genotype, key)
            return self.problem(output)

        # Create the ME initial state
        dna_sample_key, score_init_key, mpe_key = jr.split(key, 3)

        dnas = dna_distribution(self.popsize, key=dna_sample_key).reshape(self.popsize, -1)

        scores = _eval(dnas, jr.split(score_init_key, self.popsize))

        mpe_state = self.qd_algorithm.init(dnas, centroids, scores, mpe_key)

        def step_fn(carry, _):
            # jax.debug.print("map-elite iteration: {}", i)

            mpe_state, key = carry
            eval_key, next_key = jr.split(key)

            dnas = self.qd_algorithm.ask(mpe_state)
            scores = _eval(dnas, jr.split(eval_key, self.popsize))
            mpe_state, metrics = self.qd_algorithm.tell(dnas, scores, mpe_state)  # type: ignore

            return (mpe_state, next_key), (scores, metrics)

        final_state, scores_and_metrics = jax.lax.scan(
            step_fn,
            (mpe_state, key),
            jnp.arange(self.n_iters)
        )

        return scores_and_metrics, final_state

    def init_centroids(self, key):
        centroids, _ = compute_cvt_centroids(
            self.problem.descriptor_length, # type: ignore
            self.n_centroid_samples,
            self.n_centroids,
            self.problem.descriptor_min_val, # type: ignore
            self.problem.descriptor_max_val, # type: ignore
            key
        )

        return centroids

    def aggregate_metrics(self, metric_values: Dict[str, ArrayLike]):
        return metric_values


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

    @jax_partial
    def init(self, _, key):
        goals = self.goals()
        inputs = jr.normal(key, (len(goals), self.input_size))
        optax_state = self.optim.init(inputs)
        return (inputs, goals), optax_state

    @jax_partial
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

    @jax_partial
    def validate(
        self,
        model: FunctionalModel,
        state: PyTree,
        key: jr.KeyArray
    ):
        (inputs, goals), _ = state
        batched_keys = jr.split(key, len(inputs))

        pred = jax.vmap(model)(inputs, batched_keys)
        pred, goals = self.prepare_batch(pred, goals)

        metrics = self.metrics.compute(pred, goals)

        return (metrics, pred), state  # No extra results here

    @jax_partial
    def predict(
        self,
        model: FunctionalModel,
        state: PyTree,
        key: jr.KeyArray,
    ):
        inputs = state[0][0]
        batched_keys = jr.split(key, len(inputs))
        return jax.vmap(model)(inputs, batched_keys)[0], state


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

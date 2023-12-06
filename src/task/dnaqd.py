from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PyTree, ArrayLike, Float, Array
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids

# from qdax.utils.metrics import default_qd_metrics

from src.task.base import Task
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
        dna_variance_coefficient: float = 1.0,
    ) -> None:
        if n_centroid_samples is None:
            n_centroid_samples = problem.descriptor_length * n_centroids

        self.problem = problem
        self.qd_algorithm = qd_algorithm
        self.n_iters = n_iters
        self.popsize = popsize
        self.n_centroids = n_centroids
        self.n_centroid_samples = n_centroid_samples
        self.dna_variance_coefficient = dna_variance_coefficient

    @property
    def mode(self):
        return 'max'

    # @jit_method
    def init(self, stage, training_state, key):
        if stage == "train":
            return self.init_centroids(key)
        return training_state  # just return the centroids we used for initalization

    @jax_partial
    def overall_fitness(
        self,
        model_and_dna: Tuple[FunctionalModel, DNADistribution],
        metrics: Dict[str, Float[Array, "..."]],
        key
    ):
        qd_score = metrics['qd_score'][-1]
        coverage = metrics['coverage'][-1]

        # assume dna_samples are distributions over possible strings, so it has shape
        # (pop_size, string length * alphabet size)
        dna_sample = model_and_dna[1](self.popsize, key=key).reshape(self.popsize, -1)
        dna_variance = dna_sample.var()

        # coverage is on a percetange basis, ratio it. regularize over dna variance
        score = qd_score * coverage / 100 + self.dna_variance_coefficient * dna_variance

        return score

    @jax_partial
    def eval(
        self,
        model_and_dna: Tuple[FunctionalModel, DNADistribution],
        centroids: PyTree,
        key
    ):
        predict_key, fitness_key = jr.split(key)
        (_, metrics), _ = self.predict(model_and_dna, centroids, predict_key)  # type: ignore

        fitness = self.overall_fitness(model_and_dna, metrics, fitness_key)
        # total_score = self.score_to_coverage_ratio * qd_score + coverage
        # currently this is not working, introduce some term that maximizes variation amongst dnas...

        return fitness, centroids

    @jax_partial
    def validate(
        self,
        model_and_dna,
        centroids,
        key
    ):
        predict_key, fitness_key = jr.split(key)

        (_, metrics), (mpe_state, _) = self.predict(model_and_dna, centroids, key) # type: ignore

        fitness = self.overall_fitness(model_and_dna, metrics, fitness_key)

        metrics['fitness'] = fitness

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

        scores, metrics = scores_and_metrics

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

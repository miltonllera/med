from typing import Callable, Dict, Tuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jaxtyping import PyTree, ArrayLike, Float, Array

from src.problem.base import QDProblem
from src.task.base import Task
from src.model.base import FunctionalModel
from src.nn.dna import DNADistribution
from src.evo.qd import (
    MapElitesRepertoire,
    MAPElites,
    QDScoreAggregator,
    genotype_to_phenotype_pairwise_difference,
    qd_score_x_coverage,
    compute_cvt_centroids
)
from src.analysis.qd import _plot_2d_repertoire

from src.utils import jax_partial


QD_ALGORITHM = MAPElites  # Use Union to add more algorithms later


class QDSearchDNA(Task):
    """
    Search over possible DNA sequences that guide the rollout of a developmental model. Note that
    unlike other tasks, the models here are pairs of NCA + DNA generator. This can be any function
    that returns a sample of sequences, whether this is a fixed list or randomly generated.
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
        n_iters: int,
        popsize: int,
        n_centroids: int =1000,
        n_centroid_samples: Optional[int] = None,
        score_aggregator: Optional[QDScoreAggregator] = None,
        dna_variance_coefficient: float = 1.0,
    ) -> None:
        if n_centroid_samples is None:
            n_centroid_samples = problem.descriptor_length * n_centroids

        if score_aggregator is None:
            score_aggregator = QDScoreAggregator(
                qd_score_x_coverage, genotype_to_phenotype_pairwise_difference
            )

        self.problem = problem
        self.qd_algorithm = qd_algorithm
        self.n_iters = n_iters
        self.popsize = popsize
        self.n_centroids = n_centroids
        self.n_centroid_samples = n_centroid_samples  # type: ignore
        self.score_aggregator = score_aggregator
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
        genotypes_and_phenotypes: Tuple[Array, PyTree],
        metrics: Dict[str, Float[Array, "..."]],
        repertoire: MapElitesRepertoire,
        key: jr.PRNGKeyArray,
    ):
        aggregated_qd_score = self.score_aggregator(genotypes_and_phenotypes, metrics, repertoire)

        # DNA samples have shape (n_iters, pop_size, n_features)
        # compute variance across population, average over qd iterations
        dna_samples = genotypes_and_phenotypes[0]
        dna_variance = dna_samples.var(axis=(1, 2)).mean()

        individual_terms = {
            'aggregated_qd_score': aggregated_qd_score,
            'dna_variance': dna_variance,
        }

        # regularize using dna variance
        score = aggregated_qd_score + self.dna_variance_coefficient * dna_variance

        return score, individual_terms

    @jax_partial
    def eval(
        self,
        model_and_dna: Tuple[FunctionalModel, DNADistribution],
        centroids: PyTree,
        key
    ):
        genotypes_and_phenotypes, (_, metrics), (mpe_state, key) = self.predict(
            model_and_dna, centroids, key
        )  # type: ignore

        fitness, _ = self.overall_fitness(
            genotypes_and_phenotypes,
            metrics,
            mpe_state[0], # mpe_state[0] == repertoire
            key
        )  # type: ignore

        return fitness, (dict(fitness=fitness), centroids)

    @jax_partial
    def validate(
        self,
        model_and_dna,
        centroids,
        key
    ):
        genotypes_and_phenotypes, (_, metrics), (mpe_state, key) = self.predict(
            model_and_dna, centroids, key
        ) # type: ignore

        fitness, individual_terms = self.overall_fitness(
            genotypes_and_phenotypes, metrics, mpe_state, key
        ) # type: ignore

        metrics['fitness'] = fitness
        metrics =  {**metrics, **individual_terms}

        extra_results = (mpe_state[0], self.problem.descriptor_info)  # mpe_state[0] == repertoire

        return (metrics, extra_results), centroids

    @jax_partial
    def predict(self, model_and_dna, centroids, key):
        model, dna_gen = model_and_dna

        @jax.vmap
        def generate_from_dna(genotype, key):
            output, _ = model(genotype, key)
            return self.problem(output), output

        # Create the ME initial state
        dna_key, score_init_key, mpe_key = jr.split(key, 3)

        dnas = dna_gen(self.popsize, key=dna_key).reshape(self.popsize, -1)

        scores, _ = generate_from_dna(dnas, jr.split(score_init_key, self.popsize))

        mpe_state = self.qd_algorithm.init(dnas, centroids, scores, mpe_key)

        def step_fn(carry, i):
            # jax.debug.print("map-elite iteration: {}", i)

            mpe_state, key = carry
            eval_key, next_key = jr.split(key)

            dnas = self.qd_algorithm.ask(mpe_state)
            scores, outputs = generate_from_dna(dnas, jr.split(eval_key, self.popsize))
            mpe_state, metrics = self.qd_algorithm.tell(dnas, scores, mpe_state)  # type: ignore

            # debugging stuff
            # jax.debug.callback(
            #     plot_2d_repertoire_jax_callback_wrapper,
            #     i,
            #     mpe_state,
            #     self.scoring_function(metrics, mpe_state[0]),
            #     self.problem.descriptor_info
            # )

            return (mpe_state, next_key), ((dnas, outputs), (scores, metrics))

        final_state, (genotype_and_phenotypes, scores_and_metrics) = jax.lax.scan(
            step_fn,
            (mpe_state, key),
            jnp.arange(self.n_iters)
        )

        return genotype_and_phenotypes, scores_and_metrics, final_state

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


def plot_2d_repertoire_jax_callback_wrapper(i, mpe_state, scores, bd_info):
    repertoire = mpe_state

    max_idx = scores.argmax()
    best_repertoire = repertoire[max_idx]

    bd_names = tuple(bd_info.keys())
    bd_limits = tuple(zip(*bd_info.values()))

    fig, _ = _plot_2d_repertoire(best_repertoire, bd_limits, bd_names)
    fig.savefig(f"plots/debug/best_repertoire-iter{i}.png")  # type: ignore
    plt.close(fig)

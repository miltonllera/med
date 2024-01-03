import jax.numpy as jnp
import jax.random as jr
from qdax.core.map_elites import EmitterState, Emitter, MapElitesRepertoire, MAPElites as BaseME
from qdax.core.emitters.cma_opt_emitter import CMAOptimizingEmitter as CMAOptEmitterBase
# from qdax.core.emitters.cma_mega_emitter import CMAMEGAEmitter as CMAMEGAEmitterBase
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.types import Centroid, Metrics, Genotype, Fitness, Descriptor, ExtraScores, RNGKey
from typing import Any, Callable, Dict, Optional, Tuple
from jaxtyping import Array, Float

from src.utils import jit_method


SCORING_RESULTS = Tuple[Fitness, Descriptor, ExtraScores]
MPE_STATE = Tuple[MapElitesRepertoire, EmitterState, RNGKey]


def _dummy_scoring_fn(_: Genotype, k: RNGKey) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    return jnp.empty((1,)), jnp.empty((1,)), {'dummy': jnp.empty((1,))}, k


class MAPElites(BaseME):
    """
    Wrapper around qdax's MAP-Elites implementation.

    The qdax implementation does not use the ask-tell interface, which means that scoring functions
    cannot be easily customized. This is necessary if we are applying developmental models to the
    genotypes in the repertoire. However, this is easy to fix: we use the '_emit' function from the
    emitter to implement the ask and the 'add' and 'update' functions from the repertoire and the
    emitter, respectively, for the tell. we override the '__init__' to pass a dummy scoring function
    and delegate the task of providing the fitness values to a different component. Finally, we must
    also overwrite the 'init' function (NOTE this is the state init, not the module init) to pass
    the scores directly since the module does not have access to the scoring function.
    """
    def __init__(self,
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics]
    ) -> None:
        super().__init__(_dummy_scoring_fn, emitter, metrics_function)

    @jit_method
    def init(
        self,
        init_genotypes: Genotype,
        centroids: Centroid,
        fitness_and_metrics: SCORING_RESULTS,
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey]:
        fitnesses, descriptors, extra_scores = fitness_and_metrics

        # init the repertoire
        repertoire = MapElitesRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state, random_key

    @jit_method
    def ask(self, mpe_state: MPE_STATE) -> Genotype:
        repertoire, emitter_state, key = mpe_state
        return self._emitter.emit(repertoire, emitter_state, key)[0]

    @jit_method
    def tell(
        self,
        genotypes: Float[Array, "..."],
        scores: SCORING_RESULTS,
        mpe_state: MPE_STATE,
    ) -> Tuple[MPE_STATE, Metrics]:
        fitnesses, descriptors, extra_scores = scores
        repertoire, emitter_state, key = mpe_state

        # update map
        repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        _, key = jr.split(key)

        # update quality-diversity metrics with the results from the current map
        metrics = self._metrics_function(repertoire)

        return (repertoire, emitter_state, key), metrics


#---------------------------------- Emitters ---------------------------------------

class CMAOptEmitter(CMAOptEmitterBase):
    """
    Emitter used to implement CMA-based Map-Elites (i.e. CMA-ME).
    """
    def __init__(self,
        batch_size: int,
        genotype_dim: int,
        sigma_g: float,
        min_count: Optional[int] = None,
        max_count: Optional[float] = None,
        *,
        centroid_kwargs: Dict[str, Any],
    ):
        # We must create dummy centroids to initialize the qdax emitter. Most parameters do not need
        # to match the ones actually used by a QD algorithm since these centroids are only used to
        # access their number.
        dummy_centroids, _ = compute_cvt_centroids(**centroid_kwargs)  # type: ignore
        super().__init__(batch_size, genotype_dim, dummy_centroids, sigma_g, min_count, max_count)


#-------------------------------- Scoring Function ---------------------------------

SCORING_FUNCTION = Callable[[Metrics, MapElitesRepertoire], float]


def coverage_only(qd_metrics, repertoire):
    coverage = qd_metrics['coverage']

    if len(coverage.shape):
        coverage = coverage[-1]

    return coverage / 100


def qd_score_x_coverage(qd_metrics, repertoire):
    qd_score = qd_metrics['qd_score']
    coverage = qd_metrics['coverage']

    if len(qd_score.shape):
        qd_score = qd_score[-1]
        coverage = coverage[-1]

    # coverage is a percetange, normalize it to (0, 1)
    return qd_score * coverage / 100

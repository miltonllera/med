from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
from qdax.core.map_elites import EmitterState, Emitter, MapElitesRepertoire, MAPElites as BaseME
from qdax.core.emitters.cma_opt_emitter import CMAOptimizingEmitter as CMAOptEmitterBase
# from qdax.core.emitters.cma_mega_emitter import CMAMEGAEmitter as CMAMEGAEmitterBase
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.types import Centroid, Metrics, Genotype, Fitness, Descriptor, ExtraScores, RNGKey
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

def coverage_only(qd_metrics):
    coverage = qd_metrics['coverage'][-1]
    return coverage / 100


def qd_score_x_coverage(qd_metrics):
    qd_score = qd_metrics['qd_score'][-1]
    coverage = qd_metrics['coverage'][-1]

    # coverage is a percetange, normalize it to (0, 1)
    return qd_score * coverage / 100


# @struct.dataclass
# class EvoParams:
#     n_centroids: int
#     n_centroid_samples: int
#     n_descriptors: int
#     descriptor_min_val: List[float]
#     descriptor_max_val: List[float]


# @struct.dataclass
# class EvoState:
#     repertoire: MapElitesRepertoire
#     emitter_state: EmitterState
#     metrics: Metrics


# # TODO: This is a hack which I need to clean-up at some point
# class MapElitesES(BaseME):
#     """
#     Wrap the MapElites class a second time so it can be used as an EvoStrat.
#     """
#     def __init__(
#         self,
#         pholder_params: PyTree,
#         n_centroids: int,
#         n_descriptors: int,
#         descriptor_min_val: List[float],
#         descriptor_max_val: List[float],
#         emitter: Emitter,
#         metrics_function: Callable[[MapElitesRepertoire], Metrics]
#     ):
#         assert len(descriptor_min_val) == n_descriptors and len(descriptor_max_val) == n_descriptors

#         super().__init__(_dummy_scoring_fn, emitter, metrics_function)
#         self.params = pholder_params
#         self.param_reshaper = ex.ParameterReshaper(pholder_params)
#         self.n_centroids = n_centroids
#         self.n_descriptors = n_descriptors
#         self.n_descriptor_min_val = descriptor_min_val
#         self.n_descriptor_max_val = descriptor_max_val

#     @property
#     def default_params(self):
#         n_init_samples = self.n_descriptors * self.n_centroids

#         return EvoParams(
#             self.n_centroids,
#             n_init_samples,
#             self.n_descriptors,
#             self.n_descriptor_min_val,
#             self.n_descriptor_max_val,
#         )


#     @partial(jax.jit, static_argnums=(0,))
#     def initialize(self, strat_key, strategy_params: EvoParams) -> EvoState:
#         centroid_key, init_key = jr.split(strat_key)

#         centroids, _ = compute_cvt_centroids(
#             strategy_params.n_descriptors,
#             strategy_params.n_centroid_samples,
#             strategy_params.n_centroids,
#             strategy_params.descriptor_min_val,
#             strategy_params.descriptor_max_val,
#             centroid_key,
#         )

#         default_fitnesses = -jnp.inf * jnp.ones(shape=strategy_params.n_centroids)

#         genotype = self.param_reshaper.flatten(self.params)

#         # default genotypes is all 0
#         default_genotypes = jax.tree_util.tree_map(
#             lambda x: jnp.zeros(shape=(strategy_params.n_centroids,) + x.shape, dtype=x.dtype),
#             genotype,
#         )

#         # default descriptor is all zeros
#         default_descriptors = jnp.zeros_like(centroids)

#         repertoire = MapElitesRepertoire.init(
#             genotypes=default_genotypes,
#             fitnesses=default_fitnesses,
#             descriptors=default_descriptors,
#             centroids=centroids,
#             extra_scores={},
#         )

#         # get initial state of the emitter
#         emitter_state, init_key = self._emitter.init(
#             init_genotypes=default_genotypes, random_key=init_key
#         )

#         # update emitter state
#         emitter_state = self._emitter.state_update(
#             emitter_state=emitter_state,
#             repertoire=repertoire,
#             genotypes=default_genotypes,
#             fitnesses=default_fitnesses,
#             descriptors=default_descriptors,
#             extra_scores={},
#         )

#         metrics = self._metrics_function(repertoire)

#         return EvoState(repertoire, emitter_state, metrics)

#     @partial(jax.jit, static_argnums=(0,))
#     def ask(
#         self,
#         key,
#         evo_state: EvoState,
#         strategy_params: EvoParams
#     ) -> Tuple[PyTree, EvoState]:
#         params = self._emitter.emit(evo_state.repertoire, evo_state.emitter_state, key)[0]
#         x = self.param_reshaper.reshape(params)  # type: ignore
#         return x, evo_state

#     @partial(jax.jit, static_argnums=(0,))
#     def tell(
#         self,
#         x: PyTree,
#         scores: SCORING_RESULTS,
#         evo_state: EvoState,
#         strategy_params: EvoParams
#     ) -> EvoState:
#         params = self.param_reshaper.flatten(x)
#         fitnesses, descriptors, extra_scores = scores

#         # update map
#         repertoire = evo_state.repertoire.add(params, descriptors, fitnesses, extra_scores)

#         # update emitter state after scoring is made
#         emitter_state = self._emitter.state_update(
#             emitter_state=evo_state.emitter_state,
#             repertoire=repertoire,
#             genotypes=params,
#             fitnesses=fitnesses,
#             descriptors=descriptors,
#             extra_scores=extra_scores,
#         )

#         # update quality-diversity metrics with the results from the current map
#         metrics = self._metrics_function(repertoire)

#         return EvoState(repertoire, emitter_state, metrics)

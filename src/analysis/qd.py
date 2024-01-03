import jax.tree_util as jtu
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from jaxtyping import Array, Float
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.utils.plotting import plot_2d_map_elites_repertoire


def _plot_2d_repertoire(repertoire, bd_limits, bd_names):
    fig, ax = plt.subplots(figsize=(10, 10))

    _, ax = plot_2d_map_elites_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=repertoire.fitnesses,
        minval=bd_limits[0],
        maxval=bd_limits[1],
        repertoire_descriptors=repertoire.descriptors,
        ax=ax
    )

    ax.set_aspect("auto")

    ax.set_xlabel(bd_names[0])
    ax.set_ylabel(bd_names[1])

    return fig, ax


def plot_2d_repertoire_callback_wrapper(
    iter,
    metrics: Dict[str, Float[Array, "..."]],
    extra_results: Tuple[MapElitesRepertoire, Dict[str, Tuple[float, float]]],
    _
) -> Tuple[List[str], List[plt.Figure]]:
    rep, bd_info = extra_results

    bd_names, bd_limits = bd_info.items()
    # limits are in (min, max) shape, transpose to iterable of min and max.
    bd_names = tuple(bd_names)
    bd_limits = tuple(zip(*tuple(bd_limits)))

    max_idx = rep.fitnesses.argmax()
    # min_idx = repertoire.fitnesses.argmin()
    # median_idx = jnp.argsort(repertoire.fitnesses)[len(repertoire.fitness)//2]

    max_repertoire = jtu.tree_map(lambda x: x[-1][max_idx], rep)
    # min_repertoire = jtu.tree_map(lambda x: x[-1][min_idx], repertoire)
    # median_repertoire = jtu.tree_map(lambda x: x[-1][median_idx], repertoire)

    # repertoires = [max_repertoire, median_repertoire, min_repertoire]
    repertoires = [max_repertoire]
    bd_limits = jtu.tree_map(lambda x: x[-1][0], bd_limits)  # limits is also repeated...

    names = [
        f"max-repertoire_iter-{iter}",
        # f"min-repertoire_iter-{iter}",
        # f"median-repertoire_iter-{iter}",
    ]

    figures = []
    for rep in repertoires:
        fig, _ = _plot_2d_repertoire(rep, bd_limits, bd_names)
        figures.append(fig)

    return names, figures

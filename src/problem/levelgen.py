from functools import partial
from itertools import product
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse as sparse
from scipy.sparse.csgraph import shortest_path, connected_components
from jaxtyping import Array, Float

from src.problem.base import QDProblem, Fitness, Descriptor, ExtraScores


class ZeldaLevelGeneration(QDProblem):
    def __init__(
        self,
        height: int,
        width: Optional[int] = None,
        #TODO: Add more properties here
        # target_length: Optional[int] = None,
        # max_enemies: int = 5,
        # n_doors: int = 2,
    ) -> None:
        if width is None:
            width = height

        self.height = height
        self.width = width
        # self.target_length = target_length
        # self.max_enemies = max_enemies
        # self.n_doors = n_doors

    @property
    def descriptor_length(self):
        return 2

    @property
    def descriptor_min_val(self):
        return 0.0, 0.0

    @property
    def descriptor_max_val(self):
        # maximum path lengthl, maximum symmetry score
        return self.height * self.width / 2, self.height * self.width

    @property
    def score_offset(self):
        # maximum number of connected components + 100 for when there are no valid tiles
        return self.height * self.width

    @partial(jax.jit, static_argnames=("self",))
    def score(self, inputs: Float[Array, "H W"]) -> Fitness:
        """
        Computes the validity of a level by assigning a value to how well it satisfies a given set
        of constraints. This is a continuos value to allow the model to explore outside the space
        of satisfying solutions. Notice that there is no ``optimal'' level, just levels that do or
        do not satisfy the constraints.
        """
        # TODO: rewrite score and measures to only compute the adjacency matrix once
        n_connected_components = jax.pure_callback(
            compute_connected_components,
            jnp.empty((1,)),
            inputs,
            self.score_offset,
        ).squeeze()

        # add max number of connected components to ensure quality scores are positive
        return -n_connected_components + self.score_offset

    @partial(jax.jit, static_argnames=("self",))
    def compute_measures(self, inputs: Float[Array, "H W"]) -> Descriptor:
        max_path, _ = self.descriptor_min_val

        path_length = jax.pure_callback(
            longest_shortest_path,
            jnp.empty((1,), dtype=jnp.float32),
            inputs,
            max_path
        )

        symmetry = compute_simmetry(inputs, (self.height, self.width))

        return jnp.concatenate([path_length, symmetry])

    @partial(jax.jit, static_argnames=("self",))
    def extra_scores(self, _) -> ExtraScores:
        return {"dummy": jnp.empty((1,))}

    def __call__(self, inputs):
        inputs = inputs.max(axis=0)
        return super().__call__(inputs)


def compute_connected_components(int_map, max_val, non_traversible_tiles=(0,)):
    int_map = ~np.isin(int_map, non_traversible_tiles)

    if not np.any(int_map):  # if there are not valid tiles, return max possible value
        return max_val

    adj_mat = construct_adj_mat(int_map)
    n_components = connected_components(
        adj_mat,
        directed=False,
        return_labels=False
    )
    return np.asarray([n_components], dtype=np.float32)  # return a numpy array to comply with 'pure_callback'


def longest_shortest_path(int_map, max_val, non_traversible_tiles=(0,)):
    """
    Use scipy's graph algorithms to compute the longerst shortest path in a map.

    By default assumes that the empty tile is 0 and it's the only non-traversable tile. Then it
    creates an adjacency matrix from the integer map.
    """
    int_map = ~np.isin(int_map, non_traversible_tiles)

    if not np.any(int_map):  # if there are not valid tiles, return max possible value
        return max_val

    adj_mat = construct_adj_mat(int_map)
    all_dist = shortest_path(adj_mat, directed=False, return_predecessors=False).astype(np.float32)

    return (all_dist[all_dist != np.inf]).max(initial=0)[None]  # comply with expected shape


directions = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]])

def construct_adj_mat(int_map):
    height, width = int_map.shape
    # pad with dummy empty tiles:
    int_map = np.pad(int_map, ((1, 1),), 'constant', constant_values=0)

    adj = []
    for (i, j) in product(range(1, height + 1), range(1, width + 1)):
        if int_map[i, j] != 0:
            neighbors = np.asarray([[i, j]]) + directions
            for (k, l) in neighbors:
                if int_map[k, l] != 0:
                    adj.append([(i - 1) * width + (j - 1), (k - 1) * width + (l - 1), 1])

    adj = np.asarray(adj)
    shape = tuple(adj.max(initial=0, axis=0)[:2] + 1)
    coo = sparse.coo_matrix((adj[:, 2], (adj[:, 0], adj[:, 1])), shape=shape, dtype=adj.dtype)

    return coo.tocsr()


def compute_simmetry(int_map, env_shape):
    """
    Compute an aggregate of both vertical and horizontal symmetry.
    """
    result = (
        compute_vertical_symmetry(int_map, env_shape) +
        compute_horizontal_symmetry(int_map, env_shape)
    ) / 2.0

    return result[None]


def compute_horizontal_symmetry(int_map, env_shape):
    """
    Function to get the horizontal symmetry of a level
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns a symmetry float value normalized to a range of 0.0 to 1.0
    """
    max_val = env_shape[0] * env_shape[1] / 2  # for example 14*14/2=98
    m = 0

    if (int_map.shape[0] % 2) == 0:
        m = jnp.sum(
            int_map[: int_map.shape[0] // 2] == jnp.flip(int_map[int_map.shape[0] // 2 :], 0)
        ) / max_val
    else:
        m = jnp.sum(
            int_map[: int_map.shape[0] // 2] == jnp.flip(int_map[int_map.shape[0] // 2 + 1 :], 0)
        ) / max_val

    return m


def compute_vertical_symmetry(int_map, env_shape):
    """
    Function to get the vertical symmetry of a level
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns a symmetry float value normalized to a range of 0.0 to 1.0
    """
    max_val = env_shape[0] * env_shape[1] / 2
    m = 0

    if (int_map.shape[1] % 2) == 0:
        m = jnp.sum(
            int_map[:, : int_map.shape[1] // 2] == jnp.flip(int_map[:, int_map.shape[1] // 2 :], 1)
        ) / max_val
    else:
        m = jnp.sum(
            int_map[:, : int_map.shape[1] // 2] == jnp.flip(int_map[:, int_map.shape[1] // 2 + 1 :], 1)
        ) / max_val

    return m

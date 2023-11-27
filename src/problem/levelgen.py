from functools import partial
from itertools import product
from collections import deque
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.problem.base import QDProblem, Fitness, Descriptor, ExtraScores


MAX_VALUE = np.finfo(np.float32).max


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
            batched_n_islands,
            jnp.empty((1,)),
            inputs,
            vectorized=True,
        ).squeeze()

        # print(n_connected_components.shape)
        # jax.debug.print("{}", n_connected_components.shape)

        # add max number of connected components to ensure quality scores are positive
        return -n_connected_components + self.score_offset

    @partial(jax.jit, static_argnames=("self",))
    def compute_measures(self, inputs: Float[Array, "H W"]) -> Descriptor:
        path_length = jax.pure_callback(
            batched_lsp,
            jnp.empty((1,)),
            inputs,
            vectorized=True,  # this will sync across all vmaps up to this point
        )

        symmetry = compute_simmetry(inputs, (self.height, self.width))

        return jnp.concatenate([path_length, symmetry])

    @partial(jax.jit, static_argnames=("self",))
    def extra_scores(self, _) -> ExtraScores:
        return {"dummy": jnp.empty(0)}

    def __call__(self, inputs):
        # Note that because the callbacks above are using vectorization, they will sync over all
        # vmaps applied. In our case, this should be one for the trainer and one for the task, so
        # two leading batch dimensions.
        inputs = inputs.argmax(axis=2)
        return super().__call__(inputs)


directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def batched_n_islands(int_maps):
    batch_shapes = int_maps.shape[:2]
    int_maps = int_maps.reshape((-1, *int_maps.shape[2:]))

    result = np.concatenate([n_islands(im) for im in int_maps])

    return result.reshape(batch_shapes)[..., None]


def n_islands(int_map: np.ndarray, non_traversible_tiles=(0,)):
    h, w = int_map.shape
    int_map = np.isin(int_map, non_traversible_tiles)
    visited = np.zeros_like(int_map, dtype=bool)

    def in_bounds(pos):
        return 0 <= pos[0] < h and 0 <= pos[1] < w

    def visit(i, j):
        to_visit = deque()

        visited[i, j] = True
        to_visit.append((i, j))

        while len(to_visit) > 0:
            i, j = to_visit.popleft()
            for di, dj in directions:
                nb = i + di, j + dj
                if in_bounds(nb) and int_map[nb] and not visited[nb]:
                    to_visit.append(nb)
                    visited[nb] = True

    n_components = 0
    for i, j in product(range(h), range(w)):
        if int_map[i, j] and not visited[i, j]:
            visit(i, j)
            n_components += 1

    return np.asarray([n_components], dtype=np.float32)


def batched_lsp(int_maps):
    batch_shapes = int_maps.shape[:2]
    int_maps = int_maps.reshape((-1, *int_maps.shape[2:]))

    result = np.concatenate([longest_shortest_path(im) for im in int_maps])

    return result.reshape(batch_shapes)[..., None]


def longest_shortest_path(int_map, non_traversible_tiles=(0,)):
    """
    Use scipy's graph algorithms to compute the longerst shortest path in a map.

    By default assumes that the empty tile is 0 and it's the only non-traversable tile. Then it
    creates an adjacency matrix from the integer map.
    """
    # print("computing shortest path")
    h, w = int_map.shape
    int_map = ~np.isin(int_map, non_traversible_tiles)

    def in_bounds(pos):
        return 0 <= pos[0] < h and 0 <= pos[1] < w

    def bfs_shortest_path(start_pos):

        to_visit = deque()
        to_visit.append(start_pos)

        visited = np.zeros_like(int_map, dtype=bool)
        visited[start_pos] = True

        min_dist = np.zeros_like(int_map, dtype=np.float32) + MAX_VALUE
        min_dist[start_pos] = 1

        while len(to_visit) > 0:
            i, j = to_visit.popleft()
            for di, dj, in directions:
                nb = i + di, j + dj
                if in_bounds(nb) and int_map[nb] and not visited[nb]:
                    min_dist[nb] = np.minimum(min_dist[nb], min_dist[i, j] + 1)
                    to_visit.append(nb)
                    visited[nb] = True

        return min_dist

    max_path = -MAX_VALUE

    for start_pos in product(range(h), range(w)):
        min_paths = bfs_shortest_path(start_pos)
        max_path = max((min_paths * (min_paths != MAX_VALUE)).max(), max_path)

    return np.asarray([max_path], dtype=np.float32)


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

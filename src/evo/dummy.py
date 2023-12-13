from typing import Optional, Union, Tuple

import jax
import evosax as ex
import chex
from flax import struct



@struct.dataclass
class EvoState:
    members: chex.Array
    best_member: chex.Array


@struct.dataclass
class EvoParams:
    sigma_init: float
    init_min: float = -1.0
    init_max: float = 1.0


class DummyES(ex.Strategy):
    """
    A dummy strategy that does not change the members of the population.
    """
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        mean_decay: float = 0,
        sigma_init: float = 1.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs,
    ):
        super().__init__(popsize, num_dims, pholder_params, mean_decay, n_devices, **fitness_kwargs)
        self.sigma_init = sigma_init

    def initialize_strategy(self, rng: chex.PRNGKey, params: EvoParams) -> EvoState:
        """`initialize` the evolution strategy."""
        init_x = jax.random.uniform(
            rng,
            (self.popsize, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max,
        )

        return EvoState(
            members=init_x,
            best_member=init_x[0],
        )

    def ask_strategy(
        self,
        rng: chex.PRNGKey,
        state: EvoState,
        params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        return state.members, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams
    ) -> EvoState:
        return state

    @property
    def params_strategy(self):
        return EvoParams(sigma_init=self.sigma_init)

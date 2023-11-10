import jax.numpy as jnp
from jaxtyping import Float, Array


def reconstruction_loss(x: Float[Array, "..."], y: Float[Array, "..."], l: int = 2):
    return jnp.power((x - y), l).sum()

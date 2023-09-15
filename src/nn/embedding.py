import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float


class Embedding(eqx.Module):
    """
    Class that implements an embedding dictionary.

    Inputs are assumed to be a sequence of one-hot encodings of the indexes of entries
    in the dictionary. The output is the corresponding sequence of continuous embeddings.
    """
    embedding: Float[Array, "A E"]  # A: alphabet size; E: embedding dimensionality

    def __init__(self, alphabet_size: int, embedding_dim: int, key: jr.PRNGKeyArray):
        super().__init__()
        self.embedding = jr.normal(key, (alphabet_size, embedding_dim))

    @property
    def alphabet_size(self):
        return self.embedding.shape[0]

    @property
    def embedding_dim(self):
        return self.embedding.shape[1]

    @jax.named_scope("src.nn.Embedding")
    def __call__(self, inputs: Float[Array, "S A"], key: jr.PRNGKeyArray = None):
        """
        Translate a DNA string (represented as an array of one-hot encodings) into
        continuous embeddings.
        """
        return jnp.matmul(inputs, self.embedding)


class PositionEmbedding(eqx.Module):
    """
    Class that implements (learnable) position embeddings which are added to content embeddings.
    """
    position_embedding: Float[Array, "S E"]  # S: max string size; E: embedding dimensionality

    def __init__(self, max_string_size: int, embedding_dim: int, key: jr.PRNGKeyArray):
        super().__init__()
        self.position_embedding = jr.normal(key, (max_string_size, embedding_dim))

    @property
    def max_sequence_size(self):
        return self.position_embedding.shape[0]

    @property
    def embedding_dim(self):
        return self.position_embedding.shape[1]

    def __call__(self, inputs: Float[Array, "S E"], key: jr.PRNGKeyArray = None):
        return inputs + self.position_embedding[:len(inputs)]

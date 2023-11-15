from abc import ABC, abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from jaxtyping import Array, Float

from src.nn.embedding import Embedding, PositionEmbedding


class DNADecoder(eqx.Module):
    """
    Class that takes a DNA string and converts it into a continuos embedding.
    """
    embedding: Embedding
    position_embedding: PositionEmbedding
    input_is_distribution: bool = eqx.static_field()

    def __init__(
        self,
        alphabet_size: int,
        sequence_length: int,
        embedding_size: int,
        input_is_distribution: bool = False,
        *,
        key: jr.PRNGKeyArray,
    ):
        key_emb, key_pos = jr.split(key, 2)

        self.embedding = Embedding(alphabet_size, embedding_size, key_emb)
        self.position_embedding = PositionEmbedding(sequence_length, embedding_size, key_pos)
        self.input_is_distribution = input_is_distribution

    @property
    def alphabet_size(self):
        return self.embedding.alphabet_size

    @property
    def dna_seq_length(self):
        return self.position_embedding.max_sequence_size

    @property
    def dna_shape(self):
        return self.dna_seq_length, self.alphabet_size

    @property
    def input_shape(self):
        return self.dna_seq_length, self.alphabet_size

    @property
    def total_input_size(self):
        return self.dna_seq_length * self.alphabet_size

    def __call__(self, inputs: Float[Array, "S A"], key: jr.PRNGKeyArray = None):
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(*self.dna_shape)

        if self.input_is_distribution:
            idxs = inputs.argmax(1)
            inputs = jnn.one_hot(idxs, self.alphabet_size)
        return self.position_embedding(self.embedding(inputs))


class DNAContext(eqx.Module):
    """
    Implementation of cross attention for DNA decoding in Celluar Automatas.
    """
    attention: nn.MultiheadAttention

    def __init__(
        self,
        state_size: int,
        dna_emb_size: int,
        output_size: int,
        n_heads: int,
        key: jr.KeyArray
    ):
        self.attention = nn.MultiheadAttention(
            n_heads, state_size, dna_emb_size, dna_emb_size, output_size, state_size, key=key
        )

    def __call__(
        self,
        inputs: Float[Array, "C H W"],
        dna: Float[Array, "S E"],
        key: jr.KeyArray
    ) -> Float[Array, "E H W"]:
        flattened = inputs.reshape(inputs.shape[0], -1).transpose(1, 0)
        return self.attention(
            flattened, dna, dna, key=key
        ).transpose(1, 0).reshape(-1, *inputs.shape[1:])


class DNADistribution(eqx.Module, ABC):
    @abstractmethod
    def sample_dna(self, key):
        raise NotImplementedError

    def __call__(self, n_samples, *, key):
        return jax.vmap(self.sample_dna)(jr.split(key, n_samples))


class DNAIndependentSampler(DNADistribution):
    """
    Sample DNA strings by independently sampling each character in the sequence. This uses a
    normal distribution whose parameters can be fitted as part of a model.
    """
    dna_shape: Tuple[int, int]
    logits_mean: Float[Array, "S A"]
    logits_logvar: Float[Array, "S A"]

    def __init__(
        self,
        sequence_length,
        alphabet_size,
        key,
    ):
        self.dna_shape = sequence_length, alphabet_size
        self.logits_mean = jr.normal(key, shape=(sequence_length, alphabet_size))
        self.logits_logvar = jr.normal(key, shape=(sequence_length, alphabet_size))

    def sample_dna(self, key):
        std = jnp.exp(0.5  * self.logits_logvar)
        logits = self.logits_mean + std * jr.normal(key, self.dna_shape)
        return logits

    def partition(self):
        return eqx.partition(self, eqx.is_array)

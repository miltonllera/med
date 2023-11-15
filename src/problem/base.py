from abc import ABC, abstractmethod
from jaxtyping import Array, Float
from qdax.types import Fitness, Descriptor, ExtraScores


class QDProblem(ABC):
    """
    A class that provides the values of measures of interest for a given output
    """
    @property
    @abstractmethod
    def descriptor_length(self):
        return NotImplementedError

    @property
    @abstractmethod
    def descriptor_min_val(self):
        return NotImplementedError

    @property
    @abstractmethod
    def descriptor_max_val(self):
        return NotImplementedError

    @abstractmethod
    def score(self, inputs: Float[Array, "..."]) -> Fitness:
        raise NotImplementedError

    @abstractmethod
    def compute_measures(self, inputs: Float[Array, "..."]) -> Descriptor:
        raise NotImplementedError

    @abstractmethod
    def extra_scores(self, inputs: Float[Array, "..."]) -> ExtraScores:
        raise NotImplementedError

    def __call__(self, inputs):
        return self.score(inputs), self.compute_measures(inputs), self.extra_scores(inputs)

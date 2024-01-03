from abc import ABC, abstractmethod
from typing import OrderedDict, Tuple
from jaxtyping import Array, Float
from qdax.types import Fitness, Descriptor, ExtraScores


class QDProblem(ABC):
    """
    A class that provides the values of measures of interest for a given output
    """
    @property
    @abstractmethod
    def descriptor_info(self) -> OrderedDict[str, Tuple[float, ...]]:
        raise NotImplementedError

    @property
    def descriptor_names(self):
        return tuple(self.descriptor_info.keys())

    @property
    def descriptor_length(self):
        return len(self.descriptor_info)

    @property
    def descriptor_min_val(self):
        return tuple(x[0] for x in self.descriptor_info.values())

    @property
    def descriptor_max_val(self):
        return tuple(x[1] for x in self.descriptor_info.values())

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

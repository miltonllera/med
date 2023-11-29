from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import jax.numpy as jnp
from jaxtyping import Float, Array

from src.utils import jit_method
# import equinox as eqx


class MetricCollection:
    def __init__(self,
        metric_fns: List[Callable],
        metric_names: Optional[List[str]] = None,
        aggregation_mode: str ='mean',
    ) -> None:
        if metric_names is None:
            metric_names = [f'metric_{i}' for i in range(len(metric_fns))]

        if aggregation_mode == 'mean':
            aggregation_fn = jnp.mean
        elif aggregation_mode == 'sum':
            aggregation_fn = jnp.sum
        else:
            raise ValueError()

        self.metrics = metric_fns
        self.metrics_names = metric_names
        self.aggregation_fn = aggregation_fn

    def init(self):
        return jnp.zeros(len(self.metrics), dtype=jnp.float32)

    def aggregate(self, metric_values) -> Dict[str, Float[Array, "..."]]:
        kv = zip(self.metrics_names, metric_values)

        aggregated_values = []
        for n, v in kv:
            v = self.aggregation_fn(v, axis=0)
            if v.shape == ():
                v = v.item()
            aggregated_values.append((n, v))

        return dict(aggregated_values)

    @jit_method
    def compute(self, preds, targets):
        return tuple([m(preds, targets).sum() / len(targets) for m in self.metrics])


class Task(ABC):
    metrics: MetricCollection

    @abstractmethod
    def init(self, stage, state, key):
        raise NotImplementedError

    @abstractmethod
    def eval(self, model, state, key):
        raise NotImplementedError

    @abstractmethod
    def validate(self, model, state, key):
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, state, key):
        raise NotImplementedError

    def aggregate_metrics(self, metric_values):
        return self.metrics.aggregate(metric_values)

from typing import Callable, Tuple

import jax
import jax.random as jr
from jaxtyping import Float, PyTree

from src.dataset.base import DataModule
from src.task.base import MetricCollection, Task
from src.utils import TENSOR, jitted_method

# import matplotlib.pyplot as plt
# from src.analysis.visualization import to_img


# def plot_example(outputs):
#     outputs = outputs[0]
#     plt.imshow(jax.numpy.transpose(outputs[0, :4], (1, 2, 0)))
#     plt.show()


class SupervisedTask(Task):
    datamodule: DataModule
    loss_fn: Callable
    prepare_batch: Callable

    def __init__(
        self,
        datamodule,
        loss_fn,
        metrics=None,
        prepare_batch=None,
    ) -> None:
        super().__init__()

        if prepare_batch is None:
            prepare_batch = lambda x: x
        if metrics is None:
            metrics = MetricCollection([loss_fn], ['loss'])

        self.datamodule = datamodule
        self.loss_fn = loss_fn
        self.prepare_batch = prepare_batch
        self.metrics = metrics

    """
    Task that evaluates the performance of a model at predicting targets from inputs as determined
    by the provided loss function.
    """
    @jitted_method
    def init(self, stage: str, key):
        dataset = self.datamodule.init(stage, key)
        return dataset

    @jitted_method
    def eval(
        self,
        model: PyTree,
        state: PyTree,
        key: jr.KeyArray,
    ) -> Float:
        """
        Evaluate model fitness on a single batch.

        Notice that laoder is an Iterator which is not supported by jax. However we can make use
        of 'io_callback' to loade data from disk in a non-pure way. This obviously involves a
        performance hit, but is acceptable when we cannot load the full data onto memory.

        """
        (pred, y), state = self.predict(model, state, key)
        pred, y = self.prepare_batch(pred, y)
        loss = jax.vmap(self.loss_fn, in_axes=(0, 0))(pred, y)
        loss = loss.sum() / len(y)
        return loss, state

    @jitted_method
    def validate(
        self,
        model: PyTree,
        state: PyTree,
        key: jr.KeyArray,
    ) -> Float:
        (pred, y), state = self.predict(model, state, key)
        pred, y = self.prepare_batch(pred, y)
        metrics = self.metrics.compute(pred, y)
        return metrics, state

    @jitted_method
    def predict(
        self,
        model: PyTree,
        state: PyTree,
        key: jr.KeyArray,
    ) -> Tuple[Tuple[TENSOR, TENSOR], PyTree]:
        state = self.datamodule.next(state)
        x, y = state.batch
        pred = jax.vmap(model, in_axes=(0, None))(x, key)

        # jax.debug.callback(plot_example, pred)

        return (pred, y), state

    __call__ = eval

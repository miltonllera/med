import os
import os.path as osp
from typing import Any, Callable, Optional, Union

import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import equinox as eqx
import optax
from qdax.utils.plotting import plot_2d_map_elites_repertoire as _plot_2d_repertoire
# from jaxtyping import ArrayLike


class Callback:
    """
    Callbacks implementing different functionality.

    The general idea is that the callbacks define what to log (fitness values, model parameters,
    etc.) and the attached logger instance defines where/how to save the information. This could
    be just saving it to disk or a more sophisticated storage such as WandB or TensorBoard.
    """
    def attach_logger(self, logger):
        self._logger = logger

    def init(self, *_):
        pass

    def train_loop_end(self, *_):
        pass

    def validation_end(self, *_):
        pass

    def train_end(self, *_):
        pass

    def test_end(self, *_):
        pass

    # TODO: add handlers for different events?


class Checkpoint(Callback):
    def __init__(self, save_dir, file_template):
        super().__init__()
        self.save_dir = save_dir
        self.file_template = file_template

        self.init()
        self._ckpt_state = None
        self._ckpt_iter = 0
        os.makedirs(save_dir)

    @property
    def best_state(self):
        raise NotImplementedError

    def train_end(self, *_):
        if self._ckpt_state is not None:
            save_pytree(
                self._ckpt_state,
                self.save_dir,
                self.file_template.format(iteration=self._ckpt_iter)
            )


class MonitorCheckpoint(Checkpoint):
    def __init__(
        self,
        save_dir: str,
        file_template: str,
        monitor_key=None,
        mode='min',
        state_getter: Optional[Union[Callable, str, int]] = None,
    ) -> None:
        super().__init__(save_dir, file_template)

        if state_getter is not None and not isinstance(state_getter, Callable):
            getter = lambda x: x[state_getter]
        else:
            getter = lambda x: x

        self.mode = mode
        self.monitor_key = monitor_key
        self.state_getter = getter

        self._best_val = (1 if mode == "min" else -1) * np.inf

    def has_improved(self, metric):
        if self.mode == "max":
            return self._best_val < metric
        return self._best_val > metric

    @property
    def best_state(self):
        return self._ckpt_state

    def validation_end(self, iter, metric, _, state) -> Any:
        state = self.state_getter(state)

        if self.monitor_key is not None:
            try:
                metric = metric[self.monitor_key]
            except KeyError as e:
                e.add_note(f"Available keys are {metric.keys()}")

        if self.has_improved(metric):
            self._best_val = metric
            self._ckpt_state = state
            self._ckpt_iter = iter


def backprop_optimizer_state(training_state):
    return training_state[1]


def search_task_optimizer_state(training_state):
    return training_state[2][1]


class LRMonitor(Callback):
    def __init__(self, state_indexer=None, key='lr_value'):
        if state_indexer is None:
            state_indexer = backprop_optimizer_state

        self.get_state = state_indexer
        self.lr_history = []
        self.key = key

    def train_loop_end(self, iteration, _, training_state):
        opt_state: optax.OptState = self.get_state(training_state)
        try:
            lr = opt_state.hyperparams['learning_rate'].item()  # type: ignore
            self._logger.log_scalar(self.key, iteration, lr)
        except AttributeError as e:
            e.add_note("Did you forget to use 'optax.inject_hyperparams'?")


class VisualizationCallback(Callback):
    """
    wrapper class around visualization functions
    """
    def __init__(self, visualization, dataset, save_dir: str, save_prefix: str = ""):
        self.viz = visualization
        self.dataset = dataset
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        os.makedirs(save_dir)

    def test_end(self, _, training_state) -> Any:
        model = training_state[0]
        model = eqx.tree_at(lambda x: x.output_dev_states, model, True)
        self.viz(model, self.dataset, osp.join(self.save_dir, self.save_prefix))

    # def validation_end(self, iter, metric, state) -> Any:
    #     model = state[0].best_member


class QDMapVisualizer(Callback):
    def __init__(self, n_iters: int, save_dir: str, save_prefix: str = "") -> None:
        super().__init__()
        self.n_iters = n_iters
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        os.makedirs(save_dir, exist_ok=True)

    def validation_end(self, iter, metrics, extra_results, _) -> Any:
        # Note: ignore the metrics in the function signature, those are averaged. We have to
        # select one repertoire and it's corresponding metrics to plot. Right now I am just
        # selecting the first one (of the last validation step), but this should be something
        # like median, min and max.
        repertoire, bd_limits = extra_results

        repertoire = jtu.tree_map(lambda x: x[-1][0], repertoire)
        # metrics = jtu.tree_map(lambda x: x[-1][0], metrics)
        bd_limits = jtu.tree_map(lambda x: x[-1][0], bd_limits)  # limits is also repeated...

        qd_steps = jnp.arange(self.n_iters)

        fig, _ = plot_2d_repertoire(
            repertoire,
            *bd_limits
        )

        if self.save_prefix == "":
            file_name = f"repertoire_iter-{iter}"
        else:
            file_name = f"{self.save_prefix}_repertoire-{iter}"

        save_file = osp.join(self.save_dir, file_name)
        fig.savefig(save_file)


def plot_2d_repertoire(repertoire, min_bd, max_bd):
    fig, ax = plt.subplots(figsize=(10, 10))

    _, ax = _plot_2d_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=repertoire.fitnesses,
        minval=min_bd,
        maxval=max_bd,
        repertoire_descriptors=repertoire.descriptors,
        ax=ax
    )

    return fig, ax


def save_pytree(model: eqx.Module, save_folder: str, save_name: str):
    save_file = osp.join(save_folder, f"{save_name}.eqx")
    eqx.tree_serialise_leaves(save_file, model)

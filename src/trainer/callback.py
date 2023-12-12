import os
import os.path as osp
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
# import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import equinox as eqx
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from jaxtyping import Float, Array, PyTree

from src.trainer.utils import PriorityQueue, save_pytree, load_pytree
from src.analysis.visualization import plot_2d_repertoire


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

    @property
    def best_state(self):
        raise NotImplementedError

    @property
    def last_state(self):
        raise NotImplementedError

    # def train_end(self, *_):
    #     if self._ckpt_state is not None:
    #         save_pytree(
    #             self._ckpt_state,
    #             self.save_dir,
    #             self.file_template.format(iteration=self._ckpt_iter)
    #         )


class MonitorCheckpoint(Checkpoint):
    def __init__(
        self,
        save_dir: str,
        file_template: str,
        k_best: int = 1,
        mode: str = 'min',
        monitor_key: Union[Callable, int, str, None] = None,
        state_getter: Optional[Union[Callable, str, int]] = None,
    ) -> None:
        file_template = "best_ckpt-" + file_template

        super().__init__(save_dir, file_template)

        if state_getter is not None and not isinstance(state_getter, Callable):
            getter = lambda x: x[state_getter]
        elif state_getter is None:
            getter = lambda x: x
        else:
            getter = state_getter

        self.k_best = k_best
        self.mode = mode
        self.monitor_key = monitor_key
        self.state_getter = getter

        self._ckpts = PriorityQueue(k_best, [])
        self._state_template = None

    def has_improved(self, metric):
        if len(self._ckpts) < self.k_best:
            return True

        return metric > self._ckpts.lowest_priority

    def best_state(self):
        best_state_file = max(self._ckpts).item
        return load_pytree(self.save_dir, best_state_file, self._state_template)

    def init(self, model, state):
        os.makedirs(self.save_dir, exist_ok=True)

        if self.state_getter is not None:
            self.state_getter.init(model, state)
        self._state_template = self.state_getter(state)

    def validation_end(self, iter, metric, _, state) -> Any:
        state = self.state_getter(state)

        if self.monitor_key is not None:
            try:
                metric = metric[self.monitor_key]
            except KeyError as e:
                e.add_note(f"Available keys are {metric.keys()}")

        priority = metric if self.mode == "max" else -metric
        if self.has_improved(priority):
            file = self.file_template.format(iteration=iter)
            to_delete = self._ckpts.push_and_pop((priority, file))

            save_pytree(state, self.save_dir, file)
            if to_delete is not None:
                path = Path(osp.join(self.save_dir, to_delete[1]))
                path.unlink(True)


class PeriodicCheckpoint(Checkpoint):
    def __init__(
        self,
        save_dir: str,
        file_template: str,
        checkpoint_freq: int = 1,
        max_checkpoints: int = 1,
        state_getter: Optional[Union[Callable, str, int]] = None,
    ) -> None:
        file_template = "periodic_ckpt-" + file_template

        super().__init__(save_dir, file_template)

        if state_getter is not None and not isinstance(state_getter, Callable):
            getter = lambda x: x[state_getter]
        else:
            getter = lambda x: x

        self.state_getter = getter
        self.file_template = file_template
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoints = max_checkpoints
        self._ckpt_files = deque()
        self._ckpt_state = None

    @property
    def last_checpoint_state(self):
        return self._ckpt_state

    def validation_end(self, iter, metric, _, state) -> None:
        if iter % self.checkpoint_freq != 0:
            return

        state = self.state_getter(state)

        if len(self._ckpt_files) == self.max_checkpoints:
            self.delete_oldest()

        self.update_files(iter, state)

    def delete_oldest(self):
        path = Path(self._ckpt_files.popleft())
        path.unlink(True)

    def update_files(self, iter, state):
        new_file = osp.join(self.save_dir, self.file_template.format(iter))
        self._ckpt_files.append(new_file)

        save_pytree(
            state,
            self.save_dir,
            self.file_template.format(iteration=self._ckpt_iter)
        )


# def backprop_optimizer_state(training_state):
#     return training_state[1]


# def search_task_optimizer_state(training_state):
#     return training_state[2][1]


# class LRMonitor(Callback):
#     def __init__(self, state_indexer=None, key='lr_value'):
#         if state_indexer is None:
#             state_indexer = backprop_optimizer_state

#         self.get_state = state_indexer
#         self.lr_history = []
#         self.key = key

#     def train_loop_end(self, iteration, _, training_state):
#         opt_state: optax.OptState = self.get_state(training_state)
#         try:
#             lr = opt_state.hyperparams['learning_rate'].item()  # type: ignore
#             self._logger.log_scalar(self.key, iteration, lr)
#         except AttributeError as e:
#             e.add_note("Did you forget to use 'optax.inject_hyperparams'?")


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
    def __init__(self, n_iters: int, save_dir: str, save_prefix: str = "", measure_names=None) -> None:
        super().__init__()
        self.n_iters = n_iters
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        self.measure_names = [] if measure_names is None else measure_names
        os.makedirs(save_dir, exist_ok=True)

    def validation_end(
        self,
        iter,
        metrics: Dict[str, Float[Array, "..."]],
        extra_results: Tuple[MapElitesRepertoire, PyTree],
        _
    ) -> None:
        # Note: ignore the metrics in the function signature, those are averaged. We have to
        # select one repertoire and it's corresponding metrics to plot. Right now I am just
        # selecting the first one (of the last validation step), but this should be something
        # like median, min and max.
        repertoire, bd_limits = extra_results

        max_idx = repertoire.fitnesses.argmax()
        # min_idx = repertoire.fitnesses.argmin()
        # median_idx = jnp.argsort(repertoire.fitnesses)[len(repertoire.fitness)//2]

        max_repertoire = jtu.tree_map(lambda x: x[-1][max_idx], repertoire)
        # min_repertoire = jtu.tree_map(lambda x: x[-1][min_idx], repertoire)
        # median_repertoire = jtu.tree_map(lambda x: x[-1][median_idx], repertoire)

        # repertoires = [max_repertoire, median_repertoire, min_repertoire]
        repertoires = [max_repertoire]
        bd_limits = jtu.tree_map(lambda x: x[-1][0], bd_limits)  # limits is also repeated...

        for key, repertoire in zip(['max', 'min', 'median'], repertoires):
            fig, ax = plot_2d_repertoire(
                repertoire,
                *bd_limits
            )

            ax.set_xlabel(self.measure_names[0])
            ax.set_ylabel(self.measure_names[1])

            if self.save_prefix == "":
                file_name = f"{key}-repertoire_iter-{iter}"
            else:
                file_name = f"{self.save_prefix}_{key}-repertoire_{iter}"

            save_file = osp.join(self.save_dir, file_name)
            fig.savefig(save_file)
            plt.close(fig)


class QDOutputPlotter(Callback):
    def __init__(self, save_dir: str, save_prefix: str = "") -> None:
        super().__init__()
        self.save_dir = save_dir
        self.save_prefix = save_prefix

    def validation_end(self,
        iter,
        metrics: Dict[str, Float[Array, "..."]],
        extra_results: Tuple[Float[Array, "..."], MapElitesRepertoire, PyTree],
        _
    ) -> None:
        outputs, repertoire = extra_results[:2]

        # take the last one of the evlaution iters
        outputs = outputs[-1]
        # max_idx = repertoire.fitnesses.argmax()
        n_models, _, n_maps = outputs.shape[:3]

        rand_model_idx = np.random.randint(0, n_models)
        rand_map_idx = np.random.randint(0, n_maps)
        rand_map = outputs[rand_model_idx, -1, rand_map_idx]

        rand_map = np.transpose(np.asarray(rand_map), (1, 2, 0)).argmax(axis=-1).astype(np.float32)

        fig = plt.gcf()

        plt.imshow(rand_map)

        fig.savefig(osp.join(self.save_dir, f"random_map-iteration_{iter}"))
        plt.close(fig)

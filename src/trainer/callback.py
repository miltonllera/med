import os
import os.path as osp
from collections import deque
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from src.trainer.utils import PriorityQueue, save_pytree, load_pytree


CALLBACK_HOOKS = Literal[
    "train_iter_end",
    "train_end",
    "validation_iter_end",
    "validaton_end",
    "test_iter_end",
    "test_end",
]


class Callback:
    """
    Callbacks implementing different functionality.

    The general idea is that the callbacks define what to log (fitness values, model parameters,
    etc.) and the attached logger instance defines where/how to save the information. This could
    be just saving it to disk or a more sophisticated storage such as WandB or TensorBoard.
    """
    def train_start(self, *_):
        pass

    def train_iter_end(self, *_):
        pass

    def train_end(self, *_):
        pass

    def validation_start(self, *_):
        pass

    def validation_iter_end(self, *_):
        pass

    def validation_end(self, *_):
        pass

    def test_init(self, *_):
        pass

    def test_iter_end(self, *_):
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
        return metric > self._ckpts.highest_priority

    @property
    def best_state(self):
        best_state_file = max(self._ckpts).item
        return load_pytree(self.save_dir, best_state_file, self._state_template)

    def train_start(self, total_steps, model, initial_trainer_state):
        os.makedirs(self.save_dir, exist_ok=True)

        if self.state_getter is not None:
            self.state_getter.init(model, initial_trainer_state)
        self._state_template = self.state_getter(initial_trainer_state)

        # use the initial state as a sentinel
        self.update_checkpoints(0, -np.inf, initial_trainer_state)

    def validation_end(self, iter, metric, state) -> Any:
        if self.monitor_key is not None:
            try:
                metric = metric[self.monitor_key]
            except KeyError as e:
                e.add_note(f"Available keys are {metric.keys()}")

        self.update_checkpoints(iter, metric, state)

    def update_checkpoints(self, iter, metric, state):
        state = self.state_getter(state)

        # because checkpoints are stored in a min heap,
        # we want the worst model to have the highest prioriy.
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

    def validation_end(self, iter, metric, state) -> None:
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


class VisualizationCallback(Callback):
    """
    wrapper class around visualization functions
    """
    def __init__(
        self,
        visualization,
        save_dir: str,
        save_prefix: str = "",
        run_on: Union[Literal["all"], List[CALLBACK_HOOKS]] = "all",
    ):
        self.viz = visualization
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        self.run_on = run_on

        os.makedirs(save_dir)

    def train_iter_end(self, *args, **kwargs):
        self._plot("train_iter_end", *args, **kwargs)

    def train_end(self, *args, **kwargs):
        self._plot("train_iter_end", *args, **kwargs)

    def validation_iter_end(self, *args, **kwargs):
        self._plot("train_iter_end", *args, **kwargs)

    def validation_end(self, *args, **kwargs):
        self._plot("train_iter_end", *args, **kwargs)

    def test_iter_end(self, *args, **kwargs):
        self._plot("train_iter_end", *args, **kwargs)

    def test_end(self, *args, **kwargs):
        self._plot("train_iter_end", *args, **kwargs)

    def _plot(self, calling_hook: Literal[CALLBACK_HOOKS], *args, **kwargs):
        if self.run_on != "all" and calling_hook not in self.run_on:
            return

        plot_names, figures = self.viz(*args, **kwargs)

        if not isinstance(plot_names, list):
            plot_names = [plot_names]
            figures = [figures]

        for name, fig in zip(plot_names, figures):
            if self.save_prefix == "":
                file_name = name
            else:
                file_name = f"{self.save_prefix}_{name}"

            save_file = osp.join(self.save_dir, file_name)
            fig.savefig(save_file)
            plt.close(fig)

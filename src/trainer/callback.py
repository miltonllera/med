import os
import os.path as osp
from typing import Any, Callable, Optional, Union

import numpy as np
import equinox as eqx


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

    def validation_end(self, iter, metric, state) -> Any:
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

    def test_end(self, _, training_state):
        model = training_state[0]
        model = eqx.tree_at(lambda x: x.output_dev_states, model, True)
        self.viz(model, self.dataset, osp.join(self.save_dir, self.save_prefix))


def save_pytree(model: eqx.Module, save_folder: str, save_name: str):
    save_file = osp.join(save_folder, f"{save_name}.eqx")
    eqx.tree_serialise_leaves(save_file, model)

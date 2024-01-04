from abc import abstractmethod, ABC
from typing import List, Optional
from logging import getLogger

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.random import KeyArray, split as split_key

from src.trainer.callback import Callback
from src.trainer.logger import Logger
from src.task.base import Task
from src.utils import tree_stack


_logger = getLogger(__name__)


class Trainer(ABC):
    def __init__(
        self,
        task: Task,
        steps: int,
        eval_steps: int,
        eval_freq: Optional[int],
        loggers: Optional[List[Logger]],
        callbacks: Optional[List[Callback]],
        use_progress_bar: bool = True,
        # performance options
        _jit_step_fns: bool = False,  # By default, do not merge the outer step functions
    ) -> None:
        if loggers is None:
            loggers = []

        if callbacks is None:
            callbacks = []

        if len(loggers) > 0:
            for c in callbacks:
                c.attach_logger(loggers[0])

        if eval_freq is None:
            eval_freq = eval_steps

        # if use_progress_bar and

        self.task = task
        self.steps = steps
        self.eval_steps = eval_steps
        self.val_freq = eval_freq
        self.loggers = loggers
        self.callbacks = callbacks
        self.use_progress_bar = use_progress_bar
        self.metrics_formatter = lambda x: x
        self._jit_step_fns = _jit_step_fns

    @abstractmethod
    def run(self, model: eqx.Module, key: KeyArray) -> eqx.Module:
        raise NotImplementedError

    @abstractmethod
    def init(self, stage, model, state=None, *, key):
        raise NotImplementedError

    def format_results(self, stage, metric_values):
        """
        This method just adds the stage of evaluation in front of the metric names
        """
        metrics_dict = self.task.aggregate_metrics(metric_values)
        return {f"{stage}/{k}": v for (k, v) in metrics_dict.items()}

    def _fit_loop(self, model, train_step, val_step, *, key, **kwargs):
        _logger.info("Training is starting...")

        if self._jit_step_fns:
            train_step = eqx.filter_jit(train_step)
            val_step = eqx.filter_jit(val_step)

        # if self.use_progress_bar:
            # step_fn = progress_bar_scan(self.steps)(step_fn)
            # val_step_fn = progress_bar_scan(self.eval_steps)(val_step_fn)

        init_key, key = split_key(key)
        train_state = self.init("train", model, None, key=init_key, **kwargs)
        self.run_logger_and_callbacks('train_start', self.steps, model, train_state)

        for i in range(self.steps):
            train_state, fitness_or_loss = train_step(train_state, i)

            log_dict = self.format_results("train", fitness_or_loss)

            self.run_logger_and_callbacks("train_iter_end", i, log_dict, train_state)

            if ((i + 1) % self.val_freq) == 0:
                key, val_key = split_key(key)
                self._val_loop(val_step, model, train_state, val_key, **kwargs)

        self.run_logger_and_callbacks("train_end", self.steps, train_state)
        _logger.info("Training completed.")

        return train_state

    def _val_loop(self, val_step, model, trainer_state, key, **kwargs):
        val_state = self.init("val", model, trainer_state, key=key, **kwargs)

        self.run_logger_and_callbacks('validation_start', self.eval_steps, model, val_state)

        metrics_hist = []
        for i in range(self.eval_steps):
            val_state, (step_metrics, extra_results) = val_step(val_state, i)

            metrics_hist.append(step_metrics)
            log_dict = self.format_results("val", step_metrics)

            self.run_logger_and_callbacks(
                "validation_iter_end", i, log_dict, val_state, extra_results
            )

        metrics_hist = jtu.tree_map(lambda x: jnp.mean(x, axis=1), tree_stack(metrics_hist))
        metrics_hist = self.format_results("val", metrics_hist)

        self.run_logger_and_callbacks("validation_end", self.eval_steps, metrics_hist, val_state)

    def _test_loop(self, model, test_step, trainer_state, *, key, **kwargs):
        # if self.use_progress_bar:
        #     test_step = progress_bar_scan(self.eval_steps)(test_step)

        _logger.info("Test started...")

        if self._jit_step_fns:
            test_step = eqx.filter_jit(test_step)

        test_state = self.init("test", model, trainer_state, key=key, **kwargs)
        self.run_logger_and_callbacks('test_start', self.eval_steps, model, test_state)

        metrics_hist = []
        for i in range(self.eval_steps):
            test_state, (step_metrics, extra_results) = test_step(test_state, i)

            metrics_hist.append(step_metrics)
            log_dict = self.format_results("test", step_metrics)

            self.run_logger_and_callbacks("test_iter_end", i, log_dict, test_state, extra_results)

        metrics_hist = jtu.tree_map(lambda x: jnp.mean(x, axis=0), tree_stack(metrics_hist))
        metrics_hist = self.format_results("val", metrics_hist)

        self.run_logger_and_callbacks("test_end", self.eval_steps, metrics_hist, test_state)
        _logger.info("Test completed.")

    def run_logger_and_callbacks(self, hook_name: str, *args):
        if self.callbacks is not None:
            for c in self.callbacks:
                getattr(c, hook_name)(*args)

        if self.loggers is not None:
            for l in self.loggers:
                getattr(l, hook_name)(*args)

from abc import abstractmethod, ABC
from typing import List, Optional
from logging import getLogger

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.random import KeyArray, split as split_key

from src.trainer.callback import Callback
from src.trainer.logger import Logger
from src.trainer.utils import progress_bar_scan
from src.task.base import Task


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
    ) -> None:
        if loggers is None:
            # TODO: Instantiate default logger
            raise NotImplementedError

        if callbacks is not None:
            for c in callbacks:
                c.attach_logger(loggers[0])

        if eval_freq is None:
            eval_freq = eval_steps

        self.task = task
        self.steps = steps
        self.eval_steps = eval_steps
        self.eval_freq = eval_freq
        self.loggers = loggers
        self.callbacks = callbacks
        self.use_progress_bar = use_progress_bar
        self.metrics_formatter = lambda x: x

    @abstractmethod
    def run(self, model: eqx.Module, key: KeyArray) -> eqx.Module:
        raise NotImplementedError

    @abstractmethod
    def init(self, stage, model, state=None, *, key):
        raise NotImplementedError

    @abstractmethod
    def format_training_resutls(self, fitness_or_loss):
        raise NotImplementedError

    def _fit_loop(self, model, step_fn, val_step_fn, *, key, **kwargs):
        loop_iters = self.eval_freq
        n_loops = self.steps // self.eval_freq

        if self.steps % self.eval_freq != 0:
            _logger.info(
                "The number of steps in the trainer is not divisible by the eval_freq, "
                "thus more steps than specified will be run: "
                f"steps = {self.steps}, eval_freq = {self.eval_freq}"
            )

        self.run_callbacks('init')

        if self.use_progress_bar:
            step_fn = progress_bar_scan(loop_iters)(step_fn)
            val_step_fn = progress_bar_scan(self.eval_steps)(val_step_fn)

        init_key, val_key = split_key(key)
        training_state = self.init("train", model, None, key=init_key, **kwargs)

        for i in range(n_loops):
            training_state, fitness_or_loss = jax.lax.scan(
                step_fn, training_state, jnp.arange(loop_iters)
            )

            log_dict = self.format_training_resutls(fitness_or_loss)

            self.train_loop_end((i + 1) * loop_iters, log_dict, training_state)

            val_key, init_val_key = split_key(val_key)
            validation_state = self.init("val", model, training_state, key=init_val_key, **kwargs)

            validation_state, validation_metrics = jax.lax.scan(
                val_step_fn, validation_state, jnp.arange(self.eval_steps)
            )

            validation_metrics = self.format_metrics("val", validation_metrics)

            self.validation_end((i + 1) * loop_iters, validation_metrics, validation_state)

        self.train_end(n_loops * loop_iters, training_state)

        return training_state

    def _test_loop(self, model, test_fn, trainer_state, *, key, **kwargs):
        if self.use_progress_bar:
            test_fn = progress_bar_scan(self.eval_steps)(test_fn)

        test_state = self.init("test", model, trainer_state, key=key, **kwargs)

        # TODO: For some reason this isn't working and I am tired of trying to figure out why
        # UPDATE: I found that it's the 'max' operation in the max pool when jitting a combination
        # of statics and parameters, instead of wrapping the combination operation in a jitted
        # function.
        # test_state, test_metrics = jax.lax.scan(
        #     test_fn, test_state, jnp.arange(self.eval_iters)
        # )

        # test_metrics = self.format_metrics("test", test_metrics)

        self.run_callbacks("test_end", {}, test_state)

    def format_metrics(self, stage, metric_values):
        """
        This method just adds the stage of evaluation in front of the metric names
        """
        metrics_dict = self.task.metrics.aggregate(metric_values)
        return {f"{stage}/{k}": v for (k, v) in metrics_dict.items()}

    def train_loop_end(self, iteration, fitness_or_loss, state):
        steps = list(range(iteration - self.eval_freq, iteration))
        self.run_loggers('log_dict', fitness_or_loss, steps)
        self.run_callbacks("train_loop_end", steps, fitness_or_loss, state)

    def validation_end(self, iteration, metrics, state):
        self.run_loggers('log_dict', metrics, iteration)
        self.run_callbacks("validation_end", iteration, metrics, state)

    def train_end(self, iteration, state):
        self.run_callbacks("train_end", iteration, state)

    def test_end(self, iteration, metrics, state):
        self.run_loggers("finalize")
        self.run_callbacks("test_end", iteration, metrics, state)

    def run_loggers(self, log_method, *args):
        if self.loggers is None or len(self.loggers) == 0:
            return

        for l in self.loggers:
            getattr(l, log_method)(*args)

    def run_callbacks(self, callback_method: str, *args):
        if self.callbacks is None or len(self.callbacks) == 0:
            return

        for c in self.callbacks:
            getattr(c, callback_method)(*args)

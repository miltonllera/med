from typing import List, Optional

import jax.random as jr
import equinox as eqx
import optax
from jaxtyping import PyTree

from src.trainer.base import Trainer
from src.trainer.logger import Logger
from src.trainer.callback import Callback, MonitorCheckpoint
from src.model.base import FunctionalModel
from src.task.base import Task


class BackpropTrainer(Trainer):
    def __init__(
        self,
        task: Task,
        optim: optax.GradientTransformation,
        steps: int = 1000,
        eval_steps: int = 100,
        eval_freq: Optional[int] = None,
        grad_accum: int = 1,
        logger: Optional[List[Logger]] = None,
        callbacks: Optional[List[Callback]] = None,
        use_progress_bar: bool = True,
    ):
        super().__init__(
            task, steps, eval_steps, eval_freq, logger, callbacks, use_progress_bar
        )

        if grad_accum > 1:
            optim = optax.MultiSteps(optim, every_k_schedule=grad_accum)

        self.optim = optim

    def run(
        self,
        model: FunctionalModel,
        key: jr.PRNGKeyArray,
    ):
        # Define all functions here so that they are not recompiled on each call to the loops
        train_key, test_key = jr.split(key)
        statics = model.partition()[1]

        def eval_fn(params, task_state, key):
            m = eqx.combine(params, statics)
            return self.task.eval(m, task_state, key=key)

        @eqx.filter_jit
        def grad_step(carry, _):
            params, opt_state, task_state, key = carry

            key, eval_key = jr.split(key)
            (loss_value, task_state), grads = eqx.filter_value_and_grad(eval_fn, has_aux=True)(
                params, task_state, eval_key
            )

            updates, opt_state = self.optim.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)

            return (params, opt_state, task_state, key), loss_value

        def _validation_fn(params, task_state, key):
            m = eqx.combine(params, statics)
            return self.task.validate(m, task_state, key)

        @eqx.filter_jit
        def val_step(carry, _):
            params, task_state, key = carry
            key, val_key = jr.split(key)
            metrics, task_state = _validation_fn(params, task_state, val_key)
            return (params, task_state, key), metrics

        trainer_state = self._fit_loop(model, grad_step, val_step, key=train_key)

        @eqx.filter_jit
        def test_step(carry, _):
            model, task_state, key = carry
            key, test_key = jr.split(key, 2)
            metrics, task_state = self.task.validate(model, task_state, test_key)
            return (model, task_state, key), metrics

        best_parameters = self.get_best_model(trainer_state)
        best_model = model.set_parameters(best_parameters)

        self._test_loop(
            best_model,
            test_step,
            trainer_state,
            key=test_key,
        )

    def init(
        self,
        stage: str,
        model: FunctionalModel,
        trainer_state: PyTree,
        *,
        key: jr.KeyArray,
    ):
        if stage in "train":
            task_key, loop_key = jr.split(key)

            params = model.parameters()
            optim_state = self.optim.init(params)
            task_state = self.task.init("train", task_key)
            state = params, optim_state, task_state, loop_key

        elif stage == "val":
            if trainer_state is None:
                raise ValueError

            params = trainer_state[0]
            # model = model.set_parameters(params)
            task_key, loop_key = jr.split(key)
            task_state = self.task.init("val", task_key)

            state = params, task_state, loop_key

        else:
            task_key, loop_key = jr.split(key)
            task_state = self.task.init("test", task_key)

            state = model, task_state, loop_key

        return state

    def format_training_resutls(self, fitness_or_loss):
        return {'train/loss': fitness_or_loss}

    def get_best_model(self, state):
        # look for the es_state in the callbacks, if not use the last one
        if self.callbacks is not None:
            ckpt = [c for c in self.callbacks if isinstance(c, MonitorCheckpoint)]
            if len(ckpt) > 0:
                state = ckpt[0].best_state
        return state[0]

# from functools import partial
from typing import Any, Dict, List, Optional, Union, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr
import equinox as eqx
import evosax as ex
from jaxtyping import PyTree

from src.trainer.base import Trainer
from src.trainer.callback import Callback, MonitorCheckpoint
from src.trainer.logger import Logger
from src.model.base import FunctionalModel
from src.task.base import Task


class EvoTrainer(Trainer):
    """
    A trainer that usese an evolutionary strategy to fit a model to a particular task.
    """
    def __init__(
        self,
        task: Task,
        strategy: ex.Strategy,
        strategy_params: Dict[str, Any],
        steps: int = 100,
        eval_steps: int = 1,
        eval_freq: Optional[int] = None,
        n_repeats: int = 1,
        logger: Optional[List[Logger]] = None,
        callbacks: Optional[List[Callback]] = None,
        use_progress_bar: bool = True,
    ):
        if eval_freq is None:
            eval_freq = steps

        super().__init__(
            task, steps, eval_steps, eval_freq, logger, callbacks, use_progress_bar
        )

        self.strategy = strategy
        self.strategy_params = strategy_params
        self.n_repeats = n_repeats
        self._strategy_init = None

    @property
    def n_generations(self):
        return self.steps

    def run(
        self,
        model: Union[FunctionalModel, Tuple[FunctionalModel, ...]],
        key: jr.PRNGKeyArray,
    ):
        train_key, test_key = jr.split(key)

        params, statics = split_model(model)
        strategy, strategy_params = self.init_strategy(params)

        def _eval_fn(params, task_state, key):
            m = eqx.combine(params, statics)
            return self.task.eval(m, task_state, key)

        if self.n_repeats > 1:
            def eval_fn(params, task_state, key):
                fits = jax.vmap(_eval_fn, in_axes=(None, None, 0))(
                    params, task_state, key=jr.split(key, self.n_repeats)
                ) #(nrep, pop)"

                return jnp.mean(fits, axis=0) #(pop,)
        else:
            eval_fn = _eval_fn

        def evo_step(carry, _):
            es_state, task_state, key = carry
            key, ask_key, eval_key = jr.split(key, 3)

            params, es_state = strategy.ask(ask_key, es_state, strategy_params)

            fitness, task_state = jax.vmap(eval_fn, in_axes=(0, None, None))(
                params, task_state, eval_key
            )

            # vmap causes all leaves in the resulting taks_state to have an extra leading dimension
            # with pop_size number of entries. We reduce this back to a single state before the next
            # iteratation by selecting the first element of all leaves. This is valid because by we
            # do not vmap on the state variable and hence the value is repeated for all entries.
            task_state = jtu.tree_map(lambda x: x[0], task_state)

            es_state = strategy.tell(params, fitness, es_state, strategy_params)

            return (es_state, task_state, key), fitness

        def _validation_fn(params, task_state, key):
            m = eqx.combine(params, statics)
            return self.task.validate(m, task_state, key)

        def val_step(carry, _):
            es_state, task_state, key = carry
            key, val_key, ask_key = jr.split(key, 3)

            params, _ = strategy.ask(ask_key, es_state, strategy_params)

            results, task_state = jax.vmap(_validation_fn, in_axes=(0, None, None))(
                params, task_state, val_key
            )

            task_state = jtu.tree_map(lambda x: x[0], task_state)

            return (es_state, task_state, key), results

        trainer_state = self._fit_loop(
            model,
            evo_step,
            val_step,
            key=train_key,
            strategy=strategy,
            strategy_params=strategy_params
        )

        def test_step(carry, _):
            model, task_state, key = carry
            key, test_key = jr.split(key, 2)
            metrics, task_state = self.task.validate(model, task_state, test_key)
            return (model, task_state, key), metrics

        best_parameters = self.get_best_model(trainer_state, strategy)
        best_model = model.set_parameters(best_parameters)

        self._test_loop(
            best_model,
            test_step,
            trainer_state,
            key=test_key,
            strategy=strategy,
            strategy_params=strategy_params
        )

    def init(
        self,
        stage: str,
        model: FunctionalModel,
        trainer_state: PyTree,
        *,
        key: jr.KeyArray,
        strategy: ex.Strategy,
        strategy_params: ex.EvoParams
    ):
        if stage in "train":
            strat_key, task_key, loop_key = jr.split(key, 3)
            es_state = strategy.initialize(strat_key, strategy_params)
            task_state = self.task.init("train", None, task_key)
            state = es_state, task_state, loop_key

        elif stage == "val":
            if trainer_state is None:
                raise ValueError

            es_state, trainer_task_state = trainer_state[:2]
            task_key, loop_key = jr.split(key)
            task_state = self.task.init("val", trainer_task_state, task_key)

            state = es_state, task_state, loop_key

        else:
            task_key, loop_key = jr.split(key)
            task_state = self.task.init("test", task_key)

            state = model, task_state, loop_key

        return state

    def init_strategy(self, params):
        strategy = self.strategy(pholder_params=params)

        default_params = strategy.default_params
        default_params = default_params.replace(  # type: ignore [reportGeneralTypeIssue]
           **self.strategy_params
        )
        strategy_params = default_params

        return strategy, strategy_params

    def format_training_resutls(self, fitness_or_loss):
        # Unlike backprop training, 'fitness_or_loss' is over a population instead of a single
        # set of parameters. We thus convert it to several quantities of interest:
        if self.task.mode == "max":
            key = "fitness"
        else:
            key = "loss"

        return {
            f'train/{key}_min': jnp.min(fitness_or_loss, axis=1),  # currently hardcoded for minimization
            f'train/{key}_max': jnp.max(fitness_or_loss, axis=1),  # currently hardcoded for minimization
            f'train/{key}_mean': jnp.mean(fitness_or_loss, axis=1),
            f'train/{key}_var': jnp.var(fitness_or_loss, axis=1)
        }

    def format_metrics(self, stage, metrics):
        """
        Since the base method adds the stage name in front of the names, this one only needs to
        compute the population statistics of interest: mean, variance, minimum and maximum
        """
        metrics_dict_raw = super().format_metrics(stage, metrics)
        metrics_dict = {}
        for k, v in metrics_dict_raw.items():
            metrics_dict.update({
                f"{k}_min": jnp.min(v).item(),
                f"{k}_max": jnp.max(v).item(),
                f"{k}_mean": jnp.mean(v).item(),
                f"{k}_var": jnp.var(v).item(),
            })

        return metrics_dict

    def get_best_model(self, state, strategy):
        # look for the es_state in the callbacks, if not use the last one
        if self.callbacks is not None:
            ckpt = [c for c in self.callbacks if isinstance(c, MonitorCheckpoint)]
            if len(ckpt) > 0:
                state = ckpt[0].best_state
        return strategy.param_reshaper.reshape_single(state[0].best_member)


def split_model(model: Union[FunctionalModel, Tuple[FunctionalModel, ...]]) -> Tuple[PyTree, ...]:
    if isinstance(model, FunctionalModel):
        return model.partition()

    partitions = tuple(m.partition() for m in model)

    return tuple(zip(*partitions))

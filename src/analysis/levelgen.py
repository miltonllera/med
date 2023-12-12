import os
import os.path as osp
import matplotlib.pyplot as plt
# from functools import partial

import numpy as np
import jax
import jax.random as jr
from jaxtyping import PyTree

from src.trainer.base import Trainer
from .visualization import strip


class QDModelLevelGenMapViz:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        #TODO: maybe add other figure parameters...

    def __call__(self, model: PyTree, trainer: Trainer):
        key =  jr.PRNGKey(np.random.randint(2 ** 32 - 1))

        task = trainer.task

        if not hasattr(task, 'problem') or not hasattr(task, 'popsize'):
            raise ValueError(f"Task {task} does not have a 'problem' o 'popsize' attribute")

        maps = self.generate_maps(model, task.popsize, key)  # type: ignore
        scores, measures = self.score_maps(maps, task.problem)  # type: ignore

        maps = np.asarray(maps).squeeze()
        scores = np.asarray(scores).squeeze()
        measures = np.asarray(measures).squeeze()

        best_map = maps[scores.argmax()]
        best_measure_maps = maps[measures.argmax(axis=0)]

        all_maps = np.concatenate([best_map[None], best_measure_maps])
        names =  [task.problem.score_name] + task.problem.descriptor_names  # type: ignore

        fig, axes = plt.subplots(ncols=3, figsize=(30, 10))

        for name, map, ax in zip (names, all_maps, axes):
            ax.imshow(map)
            strip(ax)
            ax.set_title(name)

        os.makedirs(self.save_dir, exist_ok=True)
        fig.savefig(osp.join(self.save_dir, "generated_maps.png"))  # type: ignore
        plt.close(fig)

    def generate_maps(self, model, n_maps, key):
        dna_sample_key, prediction_key = jr.split(key)
        nca, dna_distribution = model

        @jax.jit
        def generate_maps(key):
            dna_samples = dna_distribution(n_maps, key=key)
            maps, _ = jax.vmap(nca)(dna_samples, jr.split(prediction_key, n_maps))
            return maps.argmax(1)

        return generate_maps(dna_sample_key)

    def score_maps(self, maps, problem):
        scores = jax.vmap(problem.score)(maps)
        measures = jax.vmap(problem.compute_measures)(maps)
        return scores, measures

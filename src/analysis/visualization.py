import os.path as osp
from functools import partial
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from src.dataset.base import DataModule


def training_curve_plot(fitnesses: Float[Array, "G S"], save_folder: str):
    fitnesses = np.asarray(fitnesses.mean(axis=1))

    fig = plt.gcf()
    ax = fig.gca()

    ax.plot(np.arange(len(fitnesses)), fitnesses)
    ax.set_yscale('log')

    fig.savefig(osp.join(save_folder, "training_curve"))


def growth_visualization(
    model: Callable,
    dataset: DataModule,
    save_file: str,
):
    model = model.set_inference()
    key = jr.PRNGKey(np.random.choice(2 ** 32 - 1))

    key, init_key = jr.split(key)
    batches = zip(*dataset.init("test", init_key).batch)

    for (inputs, target) in batches:
        key = jr.PRNGKey(np.random.choice(2 ** 32 - 1))

        output, gen_steps = model(inputs, key)

        output = to_img(output[jnp.newaxis])[0]
        frames = to_img(gen_steps)

        ani = generate_growth_gif(frames)
        ani.save(osp.join(save_file, "target-growth.gif"), dpi=150, writer=PillowWriter(fps=16))


def to_img(inputs, scale=2):
    clip = partial(jnp.clip, a_min=0., a_max=1.)

    def to_rgb(x):
        # assume rgb premultiplied by alpha
        rgb, a = clip(x[:3]), clip(x[3:4])
        return clip(1.0 - a + rgb)
        # rgb, a = x[:3], x[3:4]
        # return clip(1.0 - a + rgb)

    frames = jax.device_put(jax.vmap(to_rgb)(inputs), jax.devices("cpu")[0])
    frames = np.transpose(np.asarray(frames), (0, 2, 3, 1))
    frames = np.repeat(frames, scale, 1)
    frames = np.repeat(frames, scale, 2)
    return frames


def generate_fig(output: np.ndarray):
    fig = plt.figure()
    ax = plt.gca()
    strip(ax)
    plt.imshow(output, vmin=0, vmax=1)
    return fig


def generate_growth_gif(frames: np.ndarray):
    fig = plt.figure()
    ax = plt.gca()

    strip(ax)

    im = plt.imshow(frames[0], vmin=0, vmax=1)
    def animate(i):
        ax.set_title(f"Growth step: {i}")
        im.set_array(frames[i])
        return im,

    return FuncAnimation(fig, animate, interval=200, blit=True, repeat=True, frames=len(frames))


def strip(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

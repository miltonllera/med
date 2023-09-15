import io
import requests
from functools import partial
from PIL import Image
from enum import Enum
from typing import List, NamedTuple, Tuple, Optional, Union

# from numpy.random import default_rng, Generator
# from torch.utils.data import IterableDataset
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from src.dataset.base import DataModule


class Emoji(Enum):
    BANG = "💥"
    BUTTERFLY = "🦋"
    EYE = "👁"
    FISH = "🐠"
    LADYBUG = "🐞"
    PRETZEL = "🥨"
    SALAMANDER = "🦎"
    SMILEY = "😀"
    TREE = "🎄"
    WEB = "🕸"


class DataState(NamedTuple):
    batch: Tuple[Float[Array, "E"], Float[Array, "B C H W"]]
    key: Optional[jr.KeyArray] = None


class SingleEmojiDataset(DataModule):
    emoji: Float[Array, "B C H W"]
    emoji_name: str
    target_size: int
    batch_size: int
    eval_batch_size: int


    def __init__(
        self,
        emoji: Union[str, Emoji],
        target_size: int = 40,
        pad: int = 16,
        batch_size: int = 64,
        eval_batch_size: Optional[int] = 1
        # rng: Optional[Generator] = None,
    ) -> None:
        super().__init__()

        if isinstance(emoji, str):
            emoji = Emoji[emoji.upper()]

        if eval_batch_size is None:
            eval_batch_size = batch_size

        # if rng is None:
        #     rng =jnp.random.default_rng()

        emoji_image = load_emoji(emoji.value, target_size)

        self.emoji = jnp.pad(emoji_image, ((pad, pad), (pad, pad), (0, 0)), "constant")
        self.emoji_name = emoji.name
        self.target_size = target_size
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        # self.rng = rng

    def init(self, stage: str, key: jr.KeyArray):
        batch_size = self.batch_size if stage == "train" else self.eval_batch_size

        # Only one emoji, so input is fixed
        inputs = jnp.repeat(jnp.expand_dims(
            jnp.array([0], dtype=jnp.float32), axis=0), batch_size, axis=0
        )
        targets = jnp.repeat(jnp.expand_dims(self.emoji, axis=0), batch_size, axis=0)
        return DataState(batch=(inputs, jnp.transpose(targets, [0, 3, 1, 2])))  # NCHW

    def next(self, state):
        return state


class EmojiDataset(DataModule):
    emojis: Tuple[Float[Array, "B C H W"]]
    emoji_names: List[str]
    target_size: int
    batch_size: int
    eval_batch_size: int

    def __init__(
        self,
        target_size: int = 40,
        pad: int = 16,
        batch_size: int = 64,
        eval_batch_size: Optional[int] = 1
    ) -> None:
        super().__init__()

        pad_fn = partial(np.pad, pad_width=((pad, pad), (pad, pad), (0, 0)), mode="constant")
        def init_emojis(emoji):
            emoji = load_emoji(emoji.value, target_size)
            return pad_fn(emoji)

        emojis = tuple(map(init_emojis, Emoji))
        emoji_names = (e.name for e in Emoji)

        self.emojis = emojis  # type: ignore
        self.emoji_names = emoji_names  # type: ignore
        self.target_size = target_size
        self.batch_size = batch_size
        self.eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size

    @property
    def targets(self):
        return self.emojis

    def init(self, stage: str, key: jr.KeyArray):
        key, init_key = jr.split(key)
        batch = self.get_emoji(init_key)
        return DataState(batch=batch, key=key)

    def next(self, state: DataState):
        key: jr.KeyArray = state.key
        key, next_key = jr.split(key)
        batch = self.get_emoji(next_key)
        return DataState(batch=batch, key=key)

    def get_emoji(self, key, i=None):
        if i is None:
            idxs = jr.choice(key, len(self.emojis), (self.batch_size,), replace=True)
        else:
            idxs = jnp.asarray([i])

        inputs = one_hot(idxs, len(Emoji))
        targets = jnp.asarray([self.emojis[i] for i in idxs], dtype=jnp.float32)

        return inputs, jnp.transpose(targets, [0, 3, 1, 2])


def one_hot(values, max):
    b = jnp.zeros((len(values), max), dtype=jnp.float32)
    b = b.at[jnp.arange(len(values)), values].set(1.0)
    return b


def load_emoji(emoji, max_size):
    code = hex(ord(emoji))[2:].lower()
    url = f"https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u{code}.png?raw=true"
    return load_image(url, max_size)


def load_image(url, max_size=40):
    r = requests.get(url)
    img = Image.open(io.BytesIO(r.content))
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    img = np.asarray(img, dtype=np.float32) / 255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img


def make_circle_masks(n, h, w, r=None):
    x = np.linspace(-1.0, 1.0, w)[None, None, :]
    y = np.linspace(-1.0, 1.0, h)[None, :, None]
    center = np.random.uniform(-0.5, 0.5, size=[2, n, 1, 1])
    if r is None:
        r = np.random.uniform(0.1, 0.4, size=[n, 1, 1])
    x, y = (x - center[0]) / r, (y - center[1]) / r
    mask = x * x + y * y < 1.0
    return mask.astype(float)

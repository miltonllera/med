from abc import ABC, abstractmethod
from tensorboardX import SummaryWriter
from typing import Iterable, List
from jaxtyping import Float, Array

import matplotlib.pyplot as plt

from .callback import Callback


class Logger(Callback, ABC):
    def __init__(self) -> None:
        self.owner = None
        self.callbacks = None

    @abstractmethod
    def log_scalar(self, key: str, value: List, step):
        raise NotImplementedError

    @abstractmethod
    def log_dict(self, dict_to_log, step):
        raise NotImplementedError

    @abstractmethod
    def save_artifact(self, name, artifact):
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        raise NotImplementedError


class TensorboardLogger(Logger):
    """
    Wrapper around the TensorBoard SummaryWriter class.
    """
    def __init__(self, log_dir: str) -> None:
        super().__init__()
        self.summary_writer = SummaryWriter(log_dir)

    def train_iter_end(self, iter, log_dict, _):
        self.log_dict(log_dict, iter)

    def train_end(self, *_):
        self.finalize()

    def validation_end(self, iter, log_dict, _):
        self.log_dict(log_dict, iter)
        self.finalize()

    def test_end(self, iter, log_dict, _):
        self.log_dict(log_dict, iter)
        self.finalize()

    def log_scalar(self, key, value, step):
        if not isinstance(value, list):
            value = [value]
            step = [step]

        assert len(value) == len(step)

        for v,s in zip(value, step):
            if isinstance(value, Array):
                value = value.item()
            self.summary_writer.add_scalar(key, v, s)

    def log_dict(self, dict_to_log, step):
        for k, values in dict_to_log.items():
            if isinstance(values, Iterable):
                for v, s in zip(values, step):
                    self.summary_writer.add_scalar(k, v, s)
            else:
                self.summary_writer.add_scalar(k, values, step)

    def save_artifact(self, name, artifact):
        if isinstance(artifact, plt.Figure):
            self.summary_writer.add_figure(name, artifact)
        elif isinstance(artifact, Float):
            self.summary_writer.add_image(name, artifact)
        else:
            raise ValueError(f"Unrecognized type {type(artifact)} for artifact value")

    def finalize(self):
        self.summary_writer.flush()
        self.summary_writer.close()

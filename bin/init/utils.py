import logging
from typing import Dict

import numpy as np
import jax.random as jr
# from rich import get_console, table


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU
    # setup

    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical"
    )

    # for level in logging_levels:
    #     setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


# @rank_zero_only
# def extract_metrics(metrics_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
#     extracted_metrics = {}
#     for key, value in metrics_dict.items():
#         extracted_metrics[key] = value.cpu().item()
#     return extracted_metrics


# @rank_zero_only
# def print_metrics(metrics: Dict[str, float], header: str):
#     console = get_console()

#     rich_table = table.Table(header_style="bold green")
#     rich_table.add_column(header, justify="left", no_wrap=True)
#     rich_table.add_column("Value", justify="right")

#     for key, value in metrics.items():
#         rich_table.add_row(key, str(value))

#     console.print(rich_table)

def seed_everything(seed):
    np.random.seed(seed)
    rng = np.random.default_rng()
    return jr.PRNGKey(rng.choice(2 ** 32)), rng

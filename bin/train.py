from typing import Dict

import hydra
import pyrootutils
from omegaconf import DictConfig
from jax.config import config as jcfg  # type: ignore

from .init import config, utils


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,  # add to system path
    dotenv=True,      # load environment variables .env file
    cwd=True,         # change cwd to root
)


log = utils.get_logger("bin.train")


@hydra.main(
    config_path="../configs",
    config_name="train.yaml",
    version_base="1.3" ,
)
def main(cfg: DictConfig) -> Dict[str, float]:
    log.info("Run starting...")

    if cfg.disable_jit:
        jcfg.update('jax_disable_jit', True)
        log.warn("JIT compilation has been disabled for this run. Make sure you are debugging")

    jax_key, _ = utils.seed_everything(cfg.seed)

    trainer, model = config.instantiate_run(cfg)

    log.info("Starting training phase...")
    trainer.run(model, jax_key)
    log.info("Training finished.")

    # train_metrics = utils.extract_metrics(trainer.callback_metrics)
    # utils.print_metrics(train_metrics, "Training metric")

    # if cfg.test:
    #     log.info("Starting test phase...")

    #     ckpt_path = "best" if not trainer.fast_dev_run else None
    #     trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

    #     test_metrics = utils.extract_metrics(trainer.callback_metrics)
    # else:
    #     log.info("Testing was disabled for this run. Skipping...")
    #     test_metrics = {}

    # log.info("Run completed.")

    # metrics = {**train_metrics, **test_metrics}
    # return metrics
    return {}


if __name__ == "__main__":
    main()

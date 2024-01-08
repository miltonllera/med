# import sys
# import warnings
# import traceback

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


# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))

# warnings.showwarning = warn_with_traceback


@hydra.main(config_path="../configs", config_name="train.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("Run starting...")

    if cfg.disable_jit:
        jcfg.update('jax_disable_jit', True)
        log.warn("JIT compilation has been disabled for this run. Was this intentional?")

    jax_key, _ = utils.seed_everything(cfg.seed)
    trainer, model = config.instantiate_run(cfg)
    trainer.run(model, jax_key)

    log.info("Run finished.")


if __name__ == "__main__":
    main()

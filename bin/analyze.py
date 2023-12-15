import hydra
import pyrootutils
from omegaconf import DictConfig

from .init import config, utils


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,  # add to system path
    dotenv=True,      # load environment variables .env file
    cwd=True,         # change cwd to root
)


log = utils.get_logger("bin.analysis")


@hydra.main(config_path="../configs", config_name="analyze.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("Analysis starting...")

    analysis, trainer, model = config.instantiate_analysis(cfg)

    for analysis_name, analysis_module in analysis.items():
        log.info(f"Running {analysis_name} analysis...")
        analysis_module(model, trainer)

    log.info("Analysis completed")

if __name__ == "__main__":
    main()
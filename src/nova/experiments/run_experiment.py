import logging

import hydra
from omegaconf import DictConfig

from nova.experiments.runner import BenchmarkCategory, ExperimentRunner
from nova.paths import CONFIGS_DIR

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="experiments/codeagent_without_tools")
def main(config: DictConfig) -> None:
    """Main function to run the experiment."""
    runner = ExperimentRunner(config=config)
    for category in config.categories:
        logger.info(f"Running experiment for category: {category}")
        runner.benchmark_category = BenchmarkCategory(category)
        runner.run()


if __name__ == "__main__":
    main()

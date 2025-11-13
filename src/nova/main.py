import logging

import hydra
from smolagents import CodeAgent, GradioUI

from nova.experiments.runner import ExperimentRunner
from nova.paths import CONFIGS_DIR

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="experiments/codeagent_with_tools")
def main(config) -> None:
    """Main function to run the experiment."""
    runner = ExperimentRunner(config=config)
    agent_baseline = runner.get_agent()
    agent = agent_baseline.agent
    assert isinstance(agent, CodeAgent)

    # for gradio  - run first `uv add smolagents[gradio]`
    gradio_ui = GradioUI(agent, file_upload_folder="outputs/uploads", reset_agent_memory=False)
    gradio_ui.launch()

    # for simple local query
    query = "your query"
    response = agent.run(query)
    print(response)


if __name__ == "__main__":
    main()

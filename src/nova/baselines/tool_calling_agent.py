import logging

from omegaconf import DictConfig
from smolagents import AzureOpenAIServerModel, CodeAgent, ToolCallingAgent
from smolagents.tools import Tool

from nova.baselines.base import Baseline, BaselineOutputColumns, BaselineOutputType
from nova.baselines.codeagent import CodeAgentWithTools
from nova.llm.llm import SmolAgentsLLM
from nova.tools.all_tools import (
    ToolsCategory,
    get_tools_descriptions_for_category,
    get_tools_list_for_category,
)

logger = logging.getLogger(__name__)


class SimpleToolCallingAgent(Baseline[ToolCallingAgent]):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

    def get_agent_model(self) -> AzureOpenAIServerModel:
        model = SmolAgentsLLM(
            api_config=self.config.llm.api_config,
            temperature=self.config.llm.temperature,
            max_retries=self.config.llm.max_retries,
        ).get_model()
        return model

    def get_tools(self) -> list[Tool]:
        """
        Return an empty list of tools as the simple code agent does not use any tools.
        """
        return []

    def get_managed_agents(self) -> list:
        """
        Return an empty list of managed agents as the simple code agent does not manage any agents.
        """
        return []

    def get_agent_description(self) -> str:
        return self.config.agent.description

    def get_agent(self) -> ToolCallingAgent:
        agent = ToolCallingAgent(
            # identity and instructions
            name=self.baseline_type,
            description=self.get_agent_description(),
            instructions=self.config.agent.special_instructions,
            # tools and llm
            tools=self.get_tools(),
            model=self.get_agent_model(),
            managed_agents=self.get_managed_agents(),
            # metadata
            planning_interval=self.config.agent.planning_interval,
            verbosity_level=self.config.agent.verbosity_level,
            provide_run_summary=self.config.agent.provide_run_summary,
            max_steps=self.config.agent.max_steps,
        )
        logger.info(f"Initialized {self.baseline_type} with agent: {self.config.llm.model_name}")
        return agent

    def run(self, query: str) -> BaselineOutputType:
        """
        Run the baseline model with the provided query.
        Returns the model's output.
        """
        output = self.agent.run(
            task=query,
            reset=True,  # reset the agent for each question to avoid state carryover
        )
        output_dict = {
            BaselineOutputColumns.RAW_OUTPUT: output,
            BaselineOutputColumns.CODE: self.agent.memory.return_full_code(),
            BaselineOutputColumns.FULL_STEPS: self.agent.memory.get_full_steps(),
        }
        for managed_agent in self.agent.managed_agents.values():
            agent_name = managed_agent.name
            assert isinstance(managed_agent, CodeAgent), (
                f"Expected managed_agent to be CodeAgent, but got {type(managed_agent)}"
            )
            output_dict[f"managed_agents/{agent_name}/{BaselineOutputColumns.CODE}"] = (
                managed_agent.memory.return_full_code()
            )
            output_dict[f"managed_agents/{agent_name}/{BaselineOutputColumns.FULL_STEPS}"] = (
                managed_agent.memory.get_full_steps()
            )  # type: ignore
        return output_dict


class ToolCallingAgentWithTools(SimpleToolCallingAgent):
    """
    Code agent that uses all available tools.
    This is a specialized version of the SimpleToolCallingAgent that includes all tools.
    """

    def get_tools(self) -> list[Tool]:
        """
        Return all available tools for the code agent.
        """
        tools_category = ToolsCategory(self.config.tools_category)
        tools = get_tools_list_for_category(tools_category)
        logger.info(f"Using {len(tools)} tools from category {tools_category} for the agent.")
        return tools

    def get_agent_description(self) -> str:
        overal_description = self.config.agent.description
        tools_category = ToolsCategory(self.config.tools_category)
        tools_description = get_tools_descriptions_for_category(tools_category)
        description = overal_description
        if len(tools_description) > 0:
            description = (
                description
                + "\nAvailable tools:\n"
                + "\n".join([f"{tool['name']}: {tool['description']}" for tool in tools_description])
            )
        return description


class OrchestratorToolCallingAgent(SimpleToolCallingAgent):
    def get_managed_agents(self) -> list:
        """
        Return an empty list of managed agents as the simple code agent does not manage any agents.
        """
        managed_agents = []
        for managed_agent_config in self.config.managed_agents.values():
            logger.info(f"Adding managed agent: {managed_agent_config.type}")
            agent = CodeAgentWithTools(config=managed_agent_config)
            managed_agents.append(agent.get_agent())
        logger.info(f"Total managed agents: {len(managed_agents)}")
        return managed_agents

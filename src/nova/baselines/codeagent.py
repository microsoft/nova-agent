import logging
from pathlib import Path
from typing import Callable

import yaml
from omegaconf import DictConfig
from smolagents import AzureOpenAIServerModel, CodeAgent
from smolagents.agents import PromptTemplates
from smolagents.tools import Tool

from nova.baselines.base import Baseline, BaselineOutputColumns, BaselineOutputType
from nova.llm.llm import SmolAgentsLLM
from nova.tools.all_tools import (
    ToolsCategory,
    get_tools_descriptions_for_category,
    get_tools_list_for_category,
)

logger = logging.getLogger(__name__)


class SimpleCodeAgent(Baseline[CodeAgent]):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

    def get_agent_model(self) -> AzureOpenAIServerModel:
        model = SmolAgentsLLM(
            api_config=self.config.llm.api_config,
            temperature=self.config.llm.temperature,
            max_retries=self.config.llm.max_retries,
        ).get_model()
        return model

    def get_prompt_templates(self) -> PromptTemplates:
        # orchestrator system prompt
        templates_path = Path(self.config.agent.prompt_templates_path)
        if not templates_path.exists():
            raise FileNotFoundError(f"Prompt template file not found: {templates_path}")
        with open(templates_path, "r") as f:
            return PromptTemplates(**yaml.safe_load(f))

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

    def get_final_answer_checks(self) -> list[Callable]:
        """
        Return an empty list of final answer checks as the simple code agent does not perform any checks.
        """
        return []

    def get_agent_description(self) -> str:
        return self.config.agent.description

    def get_agent(self) -> CodeAgent:
        agent = CodeAgent(
            # identity and instructions
            name=self.baseline_type,
            description=self.get_agent_description(),
            instructions=self.config.agent.special_instructions,
            # tools and llm
            tools=self.get_tools(),
            model=self.get_agent_model(),
            prompt_templates=self.get_prompt_templates(),
            managed_agents=self.get_managed_agents(),
            additional_authorized_imports=self.config.libraries.datascience,
            # executor
            executor_type=self.config.agent.executor_type,
            executor_kwargs={"additional_functions": {"open": open}},
            # metadata
            planning_interval=self.config.agent.planning_interval,
            use_structured_outputs_internally=self.config.agent.use_structured_outputs_internally,
            verbosity_level=self.config.agent.verbosity_level,
            provide_run_summary=self.config.agent.provide_run_summary,
            final_answer_checks=self.get_final_answer_checks(),
            max_steps=self.config.agent.max_steps,
        )
        model_name = self.config.llm.api_config.deployment_name
        logger.info(f"Initialized agent {self.baseline_type} with model: {model_name}")
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
        return output_dict


class CodeAgentWithTools(SimpleCodeAgent):
    """
    Code agent that uses available tools.
    This is a specialized version of the SimpleCodeAgent that includes all tools.
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


class OrchestratorCodeAgent(SimpleCodeAgent):
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

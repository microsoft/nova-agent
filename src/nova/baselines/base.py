import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Generic

from omegaconf import DictConfig
from pyparsing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
BaselineOutputType = dict[str, Any]


class BaselineType(StrEnum):
    LLM_ONLY = "llm_only"
    LLM_WITH_PI = "llm_with_python_interpreter"
    LLM_WITH_PI_AML = "llm_with_pi"
    LLM_WITH_PI_AND_RETRIES = "llm_with_python_interpreter_and_retries"
    LLM_WITH_PI_AND_RETRIES_AML = "llm_with_pi_and_retries"
    CODEAGENT_WITH_RAG = "codeagent_with_rag"
    CODEAGENT_WITHOUT_TOOLS = "codeagent_without_tools"
    CODEAGENT_WITH_TOOLS = "codeagent_with_tools"
    CODEAGENT_WITH_TOOLS_AND_PLANNING = "codeagent_with_tools_and_planning"
    TOOLCALLING_AGENT_WITH_TOOLS = "toolcalling_agent_with_tools"
    ORCHESTRATOR_CODEAGENT = "orchestrator_codeagent"
    ORCHESTRATOR_TOOLCALLING_AGENT = "orchestrator_toolcalling_agent"


class BaselineOutputColumns:
    ATTEMPTS = "attempts"
    N_ATTEMPTS = "n_attempts"
    CODE = "code"
    ERROR = "error"
    RAW_OUTPUT = "raw_output"
    ANSWER_JSON = "answer_json"
    FULL_STEPS = "full_steps"


class Baseline(ABC, Generic[T]):
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.baseline_type = self.config.type
        self.agent = self.get_agent()

    @abstractmethod
    def get_agent(self) -> T:
        raise NotImplementedError("Subclasses must implement this method to return the agent instance.")

    @abstractmethod
    def run(self, query: str) -> BaselineOutputType:
        raise NotImplementedError("Subclasses must implement this method to run the agent with a query.")

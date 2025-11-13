import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from omegaconf import DictConfig
from smolagents import LocalPythonExecutor

from nova.baselines.base import Baseline, BaselineOutputColumns, BaselineOutputType
from nova.llm.llm import AzureOpenAILLM
from nova.utils.text import extract_code_from_text

logger = logging.getLogger(__name__)


class LLMOnly(Baseline[AzureChatOpenAI]):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)
        self.system_prompt = self.config.llm.system_prompt
        self.verbose = self.config.verbose

    def get_agent(self) -> AzureChatOpenAI:
        return AzureOpenAILLM(
            api_config=self.config.llm.api_config,
            temperature=self.config.llm.temperature,
            max_retries=self.config.llm.max_retries,
        ).get_model()

    def execute_llm_output(self, output: str) -> str:
        return output

    def _run_query(self, messages: list[SystemMessage | HumanMessage]) -> tuple[str, str]:
        llm_output = self.agent.invoke(messages).content
        if not isinstance(llm_output, str):
            raise ValueError("Response from the model should be a string.")
        if self.verbose:
            logger.info("LLM generation done. Now to the execution.")
        executed_output = self.execute_llm_output(llm_output)
        return llm_output, executed_output

    def run(self, query: str) -> BaselineOutputType:
        """
        Run the baseline model with the provided query.
        Returns the model's output.
        """
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=query)]
        llm_output, executed_output = self._run_query(messages)
        output_dict = {
            BaselineOutputColumns.RAW_OUTPUT: executed_output,
            BaselineOutputColumns.CODE: llm_output,
        }
        return output_dict


class LLMWithPythonInterpreter(LLMOnly):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)
        self.libraries = self.config.libraries.datascience
        self.interpreter = LocalPythonExecutor(
            additional_authorized_imports=self.libraries,  # BASE_BUILTIN_MODULES is internally loaded by the tool
            additional_functions={"open": open},
        )
        self.interpreter.send_tools(tools={})

    def execute_llm_output(self, output: str) -> str:
        code = extract_code_from_text(output)
        if code is None:
            # TODO: figure out a way to track no code errors separately
            raise ValueError(
                "No code block found in the LLM output. Ensure code is wrapped in <code> ... </code> tags."
            )
        try:
            output = self.interpreter(code_action=code).output
            return output
        except Exception as e:
            raise RuntimeError(f"Error executing code: {e}")


class LLMWithPythonInterpreterAndRetries(LLMWithPythonInterpreter):
    # the retries are for correcting code errors
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

    def run(self, query: str) -> BaselineOutputType:
        """
        Run the baseline model with the provided query for a specified number of iterations.
        Returns the final output after processing all iterations.
        """
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=query)]
        executed_output = ""
        feedback_msgs = []
        llm_output = ""
        i = 0
        for i in range(self.config.max_retries):
            try:
                llm_output, executed_output = self._run_query(messages)
                if self.verbose:
                    logger.info(f"LLM generation {i=}.")
                break  # Exit loop if successful
            except Exception as e:
                if i == self.config.max_retries - 1:
                    feedback = 'Attempt: \n\n'.join(feedback_msgs)
                    raise RuntimeError(f"Failed after {self.config.max_retries} attempts: {e} {feedback}")
                llm_output = f"Error occurred in executing LLM's code: {e}"
            feedback_msgs.append(f"Previous code output produced error:\n{llm_output}. Try again.")
            messages.append(HumanMessage(content=feedback_msgs[-1]))
        output_dict = {
            BaselineOutputColumns.RAW_OUTPUT: executed_output,
            BaselineOutputColumns.CODE: llm_output,
            BaselineOutputColumns.N_ATTEMPTS: str(i + 1),
            BaselineOutputColumns.ATTEMPTS: feedback_msgs,
        }
        return output_dict

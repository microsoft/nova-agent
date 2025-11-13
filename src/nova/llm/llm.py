import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

from langchain_openai import AzureChatOpenAI
from omegaconf import DictConfig
from smolagents import AzureOpenAIServerModel

from nova.llm.endpoint import API_KEY_ENV_VAR, AzureOpenAIEndpoint
from nova.utils.config import get_model_config

logger = logging.getLogger(__name__)
DEFAULT_CONFIG = get_model_config("default_llm")

T = TypeVar("T")


class BaseLLM(ABC, Generic[T]):
    def __init__(
        self,
        api_config: DictConfig | dict[str, Any] = DEFAULT_CONFIG,
        deployment_name: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 4,
    ) -> None:
        """Initialize the BaseLLM.

        Args:
            api_config (DictConfig | dict[str, Any], optional): API configuration. Defaults to `DEFAULT_CONFIG`.
            deployment_name (str | None, optional): Optional deployment name. If not provided, it will be taken from
                `api_config`. This is useful for easy overriding of the deployment name without needing to create a full
                config. If only the deployment_name is provided, the default config will be used. Defaults to None.
            temperature (float, optional): Sampling temperature. Defaults to 0.0.
            max_retries (int, optional): Maximum number of retries for API calls. Defaults to 4.
        """
        self.api_config = api_config if isinstance(api_config, DictConfig) else DictConfig(api_config)
        self.deployment_name = deployment_name or self.api_config.deployment_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.endpoint = AzureOpenAIEndpoint(
            url=self.api_config.base_url,
            deployment_name=self.deployment_name,
            api_version=self.api_config.api_version,
        )

    @property
    def model_temperature(self) -> float:
        is_temp_1_model = self.deployment_name.startswith("o") or self.deployment_name.startswith("gpt-5")
        if self.temperature != 1.0 and is_temp_1_model:
            logger.info(
                f"Model {self.deployment_name} can only be run with `temperature=1.0`,"
                f" overriding requested temperature {self.temperature} to 1.0"
            )
        return 1.0 if is_temp_1_model else self.temperature  # o_series must have temperature of 1.0

    def get_token_provider_if_necessary(self) -> Callable[[], str] | None:
        """If the api_key is not set, attempt to use the token provider."""
        token_provider = None
        if self.endpoint.api_key is None:
            logger.info(f"API key is not set for model {self.deployment_name}, attempting to use token provider.")
            try:
                token_provider = self.endpoint.token_provider
            except Exception as e:
                logger.error(f"Failed to set token provider for model {self.deployment_name}: {e}")
                logger.info(
                    f"Please ensure that the `{API_KEY_ENV_VAR}` environment variable is set for api_key based "
                    "authentication, or that the `endpoint.token_provider` can successfully retrieve a valid token."
                )
        return token_provider

    @abstractmethod
    def get_model(self) -> T:
        raise NotImplementedError("Subclasses must implement this method to return the model instance.")


class SmolAgentsLLM(BaseLLM[AzureOpenAIServerModel]):
    def get_model(self) -> AzureOpenAIServerModel:
        client_kwargs: dict[str, Any] = {'max_retries': self.max_retries}
        token_provider = self.get_token_provider_if_necessary()
        if token_provider:
            client_kwargs['azure_ad_token_provider'] = token_provider
        model = AzureOpenAIServerModel(
            model_id=self.deployment_name,
            azure_endpoint=self.endpoint.url,
            api_version=self.endpoint.api_version,
            api_key=self.endpoint.api_key,
            client_kwargs=client_kwargs,
            # provide model customization parameters
            temperature=self.model_temperature,
        )
        logger.info(f'Successfully initialized AzureOpenAIServerModel {self.deployment_name}')
        return model


class AzureOpenAILLM(BaseLLM[AzureChatOpenAI]):
    def get_model(self) -> AzureChatOpenAI:
        model = AzureChatOpenAI(
            azure_deployment=self.deployment_name,
            azure_endpoint=self.endpoint.url,
            # if the token provider is None, the api_key will be used. This is automatically handled by langchain-openai
            # via env variables.
            azure_ad_token_provider=self.get_token_provider_if_necessary(),
            api_version=self.endpoint.api_version,
            max_retries=self.max_retries,
            temperature=self.model_temperature,
        )
        logger.info(f'Successfully initialized AzureChatOpenAI {self.deployment_name}')
        return model

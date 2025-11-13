import os
from dataclasses import dataclass, field
from typing import Callable

from nova.llm.azure_token import get_azure_token_provider

API_KEY_ENV_VAR = "AZURE_OPENAI_API_KEY"


@dataclass(frozen=False)
class Endpoint:
    url: str
    deployment_name: str
    api_version: str = "2025-01-01-preview"


@dataclass(frozen=False)
class AzureOpenAIEndpoint(Endpoint):
    # Having the token_provider as a property ensures it's retrieved only when requested. This avoids
    # always getting the keys for all SupportedEndpoints as soon as this module is imported.
    # Currently disallows setting it directly in __init__, but could add a setter if needed.
    # See related thread: https://stackoverflow.com/q/51079503
    _token_provider: Callable[[], str] | None = field(default=None, init=False)

    @property
    def token_provider(self) -> Callable[[], str]:
        if self._token_provider is None:
            # by default, scope is set to Azure Cognitive Services
            self._token_provider = get_azure_token_provider()
        assert self._token_provider is not None
        return self._token_provider

    @property
    def api_key(self) -> str | None:
        _api_key = os.environ.get(API_KEY_ENV_VAR, None)
        return _api_key

import base64
import json
import logging
from typing import Any, Callable

from azure.identity import get_bearer_token_provider

from nova.utils.azureml import get_credential

logger = logging.getLogger(__name__)

AZURE_COGNITIVE_SERVICES = "https://cognitiveservices.azure.com"


def token_to_json(token: str) -> Any:
    """Converts an Azure access token to its underlying JSON structure.

    Args:
        token (str): The access token.

    Returns:
        The JSON object that is stored in the token.
    """
    # This is magic code to disect the token, taken from
    # https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/identity/azure-identity/azure/identity/_internal/decorators.py#L38
    base64_meta_data = token.split(".")[1].encode("utf-8") + b"=="
    json_bytes = base64.decodebytes(base64_meta_data)
    json_string = json_bytes.decode("utf-8")
    return json.loads(json_string)


def print_token_details(token: str) -> str:
    """Creates a human-readable string with details stored in the Azure token.

    Args:
        token (str): The access token.

    Returns:
        A string with information about the identity that was given the access token.
    """
    json_dict = token_to_json(token)
    NOT_PRESENT = "(not available)"
    oid = NOT_PRESENT
    upn = NOT_PRESENT
    name = NOT_PRESENT
    appid = NOT_PRESENT
    try:
        oid = json_dict["oid"]
    except Exception:
        pass
    try:
        upn = json_dict["upn"]
    except Exception:
        pass
    try:
        name = json_dict["name"]
    except Exception:
        pass
    try:
        appid = json_dict["appid"]
    except Exception:
        pass
    return f"EntraID object ID {oid}, user principal name (upn) {upn}, name {name}, appid {appid}"


def get_azure_token_provider(scope: str = AZURE_COGNITIVE_SERVICES) -> Callable[[], str]:
    """Get a token provider for Azure Cognitive Services. The bearer token provider gets authentication tokens and
    refreshes them automatically upon expiry. This is based on AzureCliCredential authentication.
    """
    credential = get_credential()
    token = credential.get_token(scope)
    logger.info(f"Credentials for {scope=}: {print_token_details(token.token)}")
    return get_bearer_token_provider(credential, scope)

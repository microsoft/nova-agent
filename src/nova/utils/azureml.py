import logging
import os

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential, ManagedIdentityCredential

logger = logging.getLogger(__name__)

ENV_AZUREML_IDENTITY_ID = "DEFAULT_IDENTITY_CLIENT_ID"
ENV_AZUREML_RUN_ID = "AZUREML_RUN_ID"
ENV_MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_MLFLOW_RUN_ID = "MLFLOW_RUN_ID"

SUSCRIPTIONS_SEGMENT = "subscriptions"
RESOURCE_GROUPS_SEGMENT = "resourceGroups"
WORKSPACES_SEGMENT = "workspaces"


def is_running_in_azure_ml() -> bool:
    """Returns True if the given run is inside of an AzureML machine, or False if it is on a machine outside AzureML."""
    return os.environ.get(ENV_AZUREML_RUN_ID, None) is not None


def get_cluster_managed_identity_credential() -> ManagedIdentityCredential:
    """Retrieves the managed identity credential for the AzureML cluster. This first retrieves the managed identity ID
    from the environment variable `ENV_AZUREML_IDENTITY_ID`, which is set by AzureML when the run is executed
    on a cluster with a managed identity. Then it creates a `ManagedIdentityCredential` object using that ID."""
    cluster_managed_identity_id = os.environ.get(ENV_AZUREML_IDENTITY_ID, None)
    if not cluster_managed_identity_id:
        raise ValueError(
            f"Environment variable {ENV_AZUREML_IDENTITY_ID} is not set, cannot get managed identity credential."
        )
    return ManagedIdentityCredential(client_id=cluster_managed_identity_id)


def get_credential() -> AzureCliCredential | ManagedIdentityCredential:
    """Returns an Azure credential object that can be used to authenticate with Azure services.
    If the code is running in AzureML, it uses the managed identity of the cluster. Otherwise, it uses the Azure CLI
    credential, which requires the user to be logged in to Azure CLI."""
    credential = AzureCliCredential() if not is_running_in_azure_ml() else get_cluster_managed_identity_credential()
    logger.info(f"Using credential: {type(credential).__name__}")
    return credential


def get_ml_client() -> MLClient:
    uri = os.environ.get(ENV_MLFLOW_TRACKING_URI, None)
    if not uri:
        raise ValueError(f"Environment variable {ENV_MLFLOW_TRACKING_URI} is not set.")
    uri_segments = uri.split("/")
    if (
        SUSCRIPTIONS_SEGMENT not in uri_segments
        or RESOURCE_GROUPS_SEGMENT not in uri_segments
        or WORKSPACES_SEGMENT not in uri_segments
    ):
        raise ValueError(f"{uri=} does not contain the required segments for Azure ML Client creation.")

    credential = get_credential()
    subscription_id = uri_segments[uri_segments.index(SUSCRIPTIONS_SEGMENT) + 1]
    resource_group_name = uri_segments[uri_segments.index(RESOURCE_GROUPS_SEGMENT) + 1]
    workspace_name = uri_segments[uri_segments.index(WORKSPACES_SEGMENT) + 1]
    logger.info(
        f"Creating MLClient with subscription_id={subscription_id}, resource_group_name={resource_group_name}, "
        f"workspace_name={workspace_name}"
    )
    client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )
    return client


def get_aml_job_id() -> str | None:
    if not is_running_in_azure_ml():
        return None
    ml_client = get_ml_client()
    mlflow_run_id = os.environ.get(ENV_MLFLOW_RUN_ID, None)
    if not mlflow_run_id:
        raise ValueError(f"Environment variable {ENV_MLFLOW_RUN_ID} is not set. Cannot get AML job ID.")
    logger.info(f"Retrieving AzureML job ID for MLflow run ID: {mlflow_run_id}")
    this_job = ml_client.jobs.get(mlflow_run_id)
    logger.info(f"AzureML job ID: {this_job.id}")
    return this_job.id

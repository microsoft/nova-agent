import json

import yaml

from nova.llm.llm import AzureOpenAILLM
from nova.paths import SUMMARY_PROMPT_PATH


def print_log(log: list[str]) -> str:
    """
    Helper function to format the log messages.
    """
    if len(log) == 0:
        return '[No log messages]'
    else:
        return ''.join(log)


def extract_info_single_image_caption(llm_response: str) -> dict[str, str]:
    try:
        # Clean the response (remove any markdown code blocks)
        cleaned_response = llm_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]  # Remove ```json
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]  # Remove ```

        # Parse JSON
        parsed_data = json.loads(cleaned_response.strip())

        # Extract the features
        features = {
            'cell_types': parsed_data.get("cell_types", ""),
            'nuclear_features': parsed_data.get("nuclear_features", ""),
            'tissue_architecture': parsed_data.get("tissue_architecture", ""),
            'organizational_pattern': parsed_data.get("organizational_pattern", ""),
            'abnormal_features': parsed_data.get("abnormal_features", ""),
            'benign_features': parsed_data.get("benign_features", ""),
            'other_notable_features': parsed_data.get("other_notable_features", ""),
        }

        return features

    except Exception as e:
        raise ValueError(f"Error parsing JSON: {e}")


def extract_info_dense_description_caption(llm_response: str) -> dict[str, list[str]]:
    try:
        # Clean the response (remove any markdown code blocks)
        cleaned_response = llm_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]  # Remove ```json
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]  # Remove ```

        # Parse JSON
        parsed_data = json.loads(cleaned_response.strip())

        # Extract the features
        features = {
            'morphological_features': parsed_data.get("morphological_features_observed", []),
            'similarities': parsed_data.get("similarities_across_images", []),
            'variations': parsed_data.get("variations_between_images", []),
        }

        return features

    except Exception as e:
        raise ValueError(f"Error parsing JSON: {e}")


def extract_info_summary_caption(llm_response: str) -> dict[str, list[str]]:
    try:
        # Clean the response (remove any markdown code blocks)
        cleaned_response = llm_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]  # Remove ```json
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]  # Remove ```

        # Parse JSON
        parsed_data = json.loads(cleaned_response.strip())

        # Extract the features
        features = {
            'one_sentence_summary': parsed_data.get("one_sentence_summary", []),
            'cluster_name': parsed_data.get("cluster_name", []),
        }

        return features

    except Exception as e:
        raise ValueError(f"Error parsing JSON: {e}")


def summary_function(
    summary_model_name: str = "o4-mini",
    summary_model_temperature: float = 1.0,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> dict:
    """
    Function to generate a summary using an LLM. The user prompt contains the query for summarization.

    Use user provided system prompt and user prompt if provided, otherwise use the default prompts from the YAML file. If either are None then an assertion error is raised.

    Args:
        summary_model_name (str): Name of the model to be used for summarization.
        summary_model_temperature (float): Temperature setting for the model.
        system_prompt (str | None): System prompt to guide the LLM.
        user_query (str | None): User query to guide the LLM.

    Returns:
        dict: A dictionary containing the summary.
    """
    summary_llm = AzureOpenAILLM(deployment_name=summary_model_name, temperature=summary_model_temperature).get_model()

    # Load the prompts and keep it ready if not provided by user
    prompt_path = SUMMARY_PROMPT_PATH
    with open(prompt_path, "r") as file:
        prompts = yaml.safe_load(file)

    system_prompt = system_prompt or prompts["system_prompt"]
    assert system_prompt, "System prompt must be provided in the YAML file or as an argument."
    user_prompt = user_prompt or prompts["user_prompt"]
    assert user_prompt, "User query must be provided in the YAML file or as an argument."

    summary_query = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_prompt},
    ]

    summary = summary_llm.invoke(summary_query).content

    asset_dict = {"summary": summary}
    return asset_dict

import base64
from pathlib import Path

import torch
import yaml
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
from PIL import Image
from smolagents import tool

from nova.llm.llm import AzureOpenAILLM
from nova.paths import (
    HIST_IMG_CLUSTER_CAPTIONS_PROMPT_PATH,
    SINGLE_HIST_IMG_CAPTIONS_PROMPT_PATH,
    SUMMARY_PROMPT_PATH,
)
from nova.utils.summarize import (
    extract_info_dense_description_caption,
    extract_info_single_image_caption,
    extract_info_summary_caption,
    print_log,
    summary_function,
)

SUPPORTED_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "o4-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-chat",
]
DEFAULT_MODEL = "gpt-4.1"


@tool
def caption_single_histology_image_tool(
    path_to_img: str,
    system_prompt: str | None = None,
    user_query: str | None = None,
    histology_img_caption_model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> dict:
    """
    Generate a dense morphological caption for a single histology image.

    It is recommended to use the default system prompt as it provides a strong foundation for the captioning task.
    But if the user wants to use a custom system prompt to guide the behavior of the captioning model, they can provide it as an argument.

    Operations:
        - Setup captioning LLM with the specified model and temperature.
        - Define the system prompt and user query for the LLM. If these are not provided, use the default prompts.
        - Load the image from the provided path and encode it as base64.
        - Invoke the language model to generate a detailed caption for the image.

    Returns:
        dict: Dictionary containing:
            - "histology_image_description": Dictionary with keys:
                - 'cell_types': Description of cell types observed in the image.
                - 'nuclear_features': Description of nuclear features observed in the image.
                - 'tissue_architecture': Description of tissue architecture observed in the image.
                - 'organizational_pattern': Description of organizational patterns observed in the image.
                - 'abnormal_features': Description of any abnormal features observed in the image.
                - 'benign_features': Description of any benign features observed in the image.
                - 'other_notable_features': Any other notable features observed in the image.
            - 'operation_log': A log of operations performed, useful for debugging.

    Args:
        path_to_img: Path to the histology image file.
        system_prompt: Optional system prompt to set the context for the LLM. Defaults to None.
        user_query: Optional user query to include in the prompt. Defaults to None.
        histology_img_caption_model: Name/configuration for the captioning LLM. Defaults to "gpt-4.1".
        temperature: Temperature for LLM sampling. Defaults to 0.0.
    """
    assert histology_img_caption_model in SUPPORTED_MODELS, (
        f"Invalid histology_img_caption_model: {histology_img_caption_model}. Supported models are {SUPPORTED_MODELS}."
    )

    log = []

    try:
        caption_llm = AzureOpenAILLM(deployment_name=histology_img_caption_model, temperature=temperature).get_model()
        log.append(f"\n Using LLM model: {histology_img_caption_model} with temperature: {temperature}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize LLM with model {histology_img_caption_model}: {e}Operation log: {print_log(log)}"
        )

    # Load and encode image as base64
    try:
        img_path = Path(path_to_img)
        img_bytes = img_path.read_bytes()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        log.append(f"\n Encoded image {img_path.name} to base64.")
    except Exception as e:
        raise RuntimeError(f"Failed to read or encode image: {e} Operation log: {print_log(log)}")

    # Load the default prompt YAML
    with open(SINGLE_HIST_IMG_CAPTIONS_PROMPT_PATH, "r") as file:
        prompts = yaml.safe_load(file)

    # define the system and user prompts
    system_prompt = system_prompt or prompts["system_prompt"]
    user_query = user_query or prompts["user_prompt"]

    # Construct LLM query
    caption_query = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{img_base64}"}},
            ],
        },
    ]
    log.append(f"\n Constructed LLM query for image {img_path.name}.")

    # Generate dense caption
    try:
        histology_image_caption = str(caption_llm.invoke(caption_query).content)
        histology_image_description = extract_info_single_image_caption(histology_image_caption)
        log.append("\n Generated dense description from LLM.")
    except Exception as e:
        raise RuntimeError(f"Failed to generate dense description: {e}Operation log: {print(log)}")

    asset_dict = {
        "histology_image_description": histology_image_description,
        'operation_log': print_log(log),
    }
    return asset_dict


@tool
def caption_and_summarize_set_of_histology_images_tool(
    list_of_img_paths: list[str] | None = None,
    list_of_base64_images: list[str] | None = None,
    caption_system_prompt: str | None = None,
    caption_user_prompt: str | None = None,
    dense_caption_model_name: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    summary: bool = True,
    summary_model_name: str = DEFAULT_MODEL,
    summary_model_temperature: float = 1.0,
    summary_system_prompt: str | None = None,
    summary_user_prompt: str | None = None,
) -> dict[str, str | list[str] | None]:
    """
    Generate dense morphological captions for a set of histology images.
    Accepts either a list of image file paths or a list of base64-encoded image strings (but not both).
    Optionally summarizes these dense descriptions into concise summaries and provide a name for cluster of images.

    This tool does not caption each image individually, but rather generates a comprehensive description for all images together.
    If summary is requested then it generates a concise 1 sentence summary of the dense description as well as a 2-3 words cluster name for the list of images.

    Operations:
        - Load images from provided paths and encode them as base64.
        - Load a YAML prompt template or use a default one. This is for dense captioning.
        - Construct a prompt for the language model, including the system and user prompts.
        - Invoke the language model to generate a detailed caption for all images considered together.
        - If requested, generate a summary of the dense description using a separate language model.
        - If providing custom summary_user_prompt, it must be formatted to include the placeholder '{dense_description}' to insert the dense description.

    Prerequisites:
        - The provided image paths must be valid and accessible.
        - The prompt YAML file should contain 'system_prompt' and 'user_prompt' keys.
        - The summary model should be capable of generating concise summaries.

    Returns:
        dict: Dictionary containing:
            - "dense_description": dictionary with keys 'morphological_features', 'similarities', 'variations' which have list[str] as values.
            - "one_sentence_summary": A concise one-sentence summary of the dense description, if summary requested.
            - "cluster_name": A short name for the list of images, if summary requested.
            - 'operation_log': A log of operations performed, useful for debugging.

    Args:
        list_of_img_paths: List of paths to histology image files (optional).
        list_of_base64_images: List of base64-encoded images (optional).
        caption_system_prompt: System prompt for the captioning LLM. Defaults to None, uses default from YAML if not provided.
        caption_user_prompt: User prompt for the captioning LLM. Defaults to None, uses default from YAML if not provided.
        prompt_path: Path to a YAML prompt template. Uses default if None.
        dense_caption_model_name: Name of captioning LLM.
        temperature: LLM sampling temperature.
        summary: Whether to generate a summary.
        summary_model_name: Model name for summary LLM.
        summary_model_temperature: Temperature for summary model.
        summary_system_prompt: System prompt for the summary LLM. Defaults to None, uses default from YAML if not provided.
        summary_user_prompt: User prompt for the summary LLM. Defaults to None, uses default from YAML if not provided.
    """
    assert dense_caption_model_name in SUPPORTED_MODELS, (
        f"Invalid dense_caption_model_name: {dense_caption_model_name}."
    )
    assert summary_model_name in SUPPORTED_MODELS, f"Invalid summary_model_name: {summary_model_name}."

    if (list_of_img_paths is None and list_of_base64_images is None) or (
        list_of_img_paths is not None and list_of_base64_images is not None
    ):
        raise ValueError("You must provide either list_of_img_paths OR list_of_base64_images (but not both).")

    log = []

    try:
        caption_llm = AzureOpenAILLM(deployment_name=dense_caption_model_name, temperature=temperature).get_model()
        log.append(f"\n Using LLM model: {dense_caption_model_name} with temperature: {temperature}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize LLM with model {dense_caption_model_name}: {e}Operation log: {print(log)}"
        )

    # Load/encode images as base64
    try:
        if list_of_img_paths is not None:
            image_base64_list = []
            for img_path in sorted([Path(img_path) for img_path in list_of_img_paths]):
                img_bytes = img_path.read_bytes()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                image_base64_list.append(img_base64)
                log.append(f"\n Encoded image {img_path.name} to base64.")
        else:
            assert list_of_base64_images is not None, (
                "list_of_base64_images must be provided if list_of_img_paths is not."
            )
            image_base64_list = list_of_base64_images
            for idx in range(len(image_base64_list)):
                log.append(f"\n Using provided base64 image #{idx + 1}.")

    except Exception as e:
        raise RuntimeError(f"Failed to read or encode imagesOriginal error: {e}Operation log: {print(log)}")

    # Load prompt YAML
    with open(HIST_IMG_CLUSTER_CAPTIONS_PROMPT_PATH, "r") as file:
        prompts = yaml.safe_load(file)

    caption_system_prompt = caption_system_prompt or prompts["system_prompt"]
    assert caption_system_prompt, "Caption system prompt must be provided in the YAML file or as an argument."
    caption_user_prompt = caption_user_prompt or prompts["user_prompt"]
    assert caption_user_prompt, "Caption user prompt must be provided in the YAML file or as an argument."

    # Construct LLM query
    caption_query = [
        {
            "role": "system",
            "content": caption_system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": caption_user_prompt},
                *[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{img_b64}"}}
                    for img_b64 in image_base64_list
                ],
            ],
        },
    ]
    log.append(f"\n Constructed LLM query with {len(image_base64_list)} images.")

    # Generate dense caption
    try:
        dense_description = str(caption_llm.invoke(caption_query).content)
        # dense_description will be a dict with keys 'morphological_features', 'similarities', 'variations' which have list[str] as values
        dense_description = extract_info_dense_description_caption(dense_description)
        log.append("\n Generated dense description from LLM.")
    except Exception as e:
        raise RuntimeError(f"Failed to generate dense descriptionOriginal error: {e}Operation log: {print(log)}")

    # Generate summary if requested
    if summary:
        try:
            # Load prompt YAML
            with open(SUMMARY_PROMPT_PATH, "r") as file:
                prompts = yaml.safe_load(file)

            summary_system_prompt = summary_system_prompt or prompts["system_prompt"]
            assert summary_system_prompt, "Summary system prompt must be provided in the YAML file or as an argument."
            summary_user_prompt = summary_user_prompt or prompts["user_prompt"]
            assert summary_user_prompt, "Summary user prompt must be provided in the YAML file or as an argument."
            assert '{dense_description}' in summary_user_prompt, (
                "Summary user prompt must contain the placeholder '{dense_description}' to insert the dense description."
            )
            summary_user_prompt = summary_user_prompt.format(dense_description=dense_description)

            summary_text = summary_function(
                summary_model_name=summary_model_name,
                summary_model_temperature=summary_model_temperature,
                system_prompt=summary_system_prompt,
                user_prompt=summary_user_prompt,
            )["summary"]

            summary_info = extract_info_summary_caption(summary_text)
            one_sentence_summary = summary_info.get("one_sentence_summary", [])
            cluster_name = summary_info.get("cluster_name", [])

        except Exception as e:
            raise RuntimeError(f"Failed to generate summary. Original error: {e} Operation log: {print(log)}")

    else:
        one_sentence_summary, cluster_name = None, None

    log.append(
        "One-sentence summary: {}".format(one_sentence_summary)
        if one_sentence_summary
        else "No one-sentence summary generated.",
    )
    log.append("Cluster name: {}".format(cluster_name) if cluster_name else "No cluster name generated.")

    asset_dict = {
        "dense_description": dense_description,
        "one_sentence_summary": one_sentence_summary,
        "cluster_name": cluster_name,
        'operation_log': print_log(log),
    }
    return asset_dict


@tool
def score_single_histology_image_using_text_tool(
    image_path: str,
    classes: list[str],
    device: str | None = None,
    apply_softmax: bool = True,
    prompts: list[str] | None = None,
) -> dict:
    """
    Takes in a histology image (not a WSI) and a list of terms (these can be terms, prompts, classes, etc.).
    The function returns a score for each term based on how well it matches the histology image.
    If prompts are provided, then they will be used to score the image against the classes (len(prompts) == len(classes)).
        If not provided then prompts = f'An H&E image of {cls}' for each class.

    Operations:
        - Load the conch pre-trained model for histology image captioning.
        - Preprocess the image and encode it into a tensor.
        - Tokenize the provided classes into a format suitable for the model.
        - Compute similarity scores between the image and each class.
        - Return the predicted class with the highest score and normalized similarity scores for each class.

    Prerequisites:
        - The image must be a valid histology image file.
        - The classes must be provided as a list of strings with at least two entries.
        - Permission to download conch model from huggingface

    Returns:
        dict: Dictionary containing:
            - "predicted_class" (str): The class with the highest similarity score.
            - "similarity_scores" (list): Normalized similarity scores for each class.
            - 'operation_log': A log of operations performed, useful for debugging.

    Args:
        image_path: Path to the histology image file (JPG, PNG, etc.). CANNOT be a WSI (SVS, TIF, etc.).
        classes: List of classes or terms to score against the image.
        device: Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        apply_softmax: Whether to apply softmax to the similarity scores. Defaults to True.
        prompts: Optional list of custom prompts corresponding to each class. Defaults to None. If None, uses a default prompt format of 'An H&E image of {cls}' for each class.

    """

    assert len(classes) > 1, (
        "At least two classes must be provided for scoring. Otherwise the return score will be meaningless."
    )

    log = []
    log.append(f"Scoring histology image: {image_path} against classes: {', '.join(classes)}")

    # get the model ready
    model_cfg = 'conch_ViT-B-16'
    try:
        device = torch.device(device or 'cuda' if torch.cuda.is_available() else 'cpu')  # type: ignore
        model, preprocess = create_model_from_pretrained(  # type: ignore
            model_cfg=model_cfg,
            checkpoint_path="hf_hub:MahmoodLab/conch",
            device=device,  # type: ignore
        )
        _ = model.eval()
        log.append(f"Model {model_cfg} loaded and set to evaluation mode on device: {device}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_cfg}: {e}Operation log: {print(log)}")

    # Load image and process
    try:
        image = Image.open(image_path)
        image_tensor = preprocess(image).unsqueeze(0).to(device)  # type: ignore
        image = image.resize((224, 224))
        log.append(f"Image {image_path} loaded and preprocessed.")
    except Exception as e:
        raise RuntimeError(f"Failed to load or preprocess image {image_path}: {e}Operation log: {print(log)}")

    # Get tokenizer and prompts
    try:
        tokenizer = get_tokenizer()
        if prompts:
            assert len(prompts) == len(classes), (
                f"Number of prompts ({len(prompts)}) must match number of classes ({len(classes)})."
            )
        prompts = prompts or [f'An H&E image of {cls}' for cls in classes]
        tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer).to(device)
        log.append(f"Tokenized prompts for classes: {', '.join(prompts)}")
    except Exception as e:
        raise RuntimeError(f"Failed to tokenize prompts: {e}Operation log: {print(log)}")

    # Compute similarity scores
    try:
        with torch.inference_mode():
            image_embeddings = model.encode_image(image_tensor)
            text_embeddings = model.encode_text(tokenized_prompts)

            if apply_softmax:
                sim_scores = (
                    (image_embeddings @ text_embeddings.T * model.logit_scale.exp()).softmax(dim=-1).cpu().numpy()
                )
            else:
                sim_scores = (image_embeddings @ text_embeddings.T * model.logit_scale.exp()).cpu().numpy()

            predicted_class = classes[sim_scores.argmax()]
            norm_sim_scores_per_class = [f"{cls}: {score:.3f}" for cls, score in zip(classes, sim_scores[0])]

        log.append(
            f"Computed similarity scores for image {image_path} against classes: {', '.join(norm_sim_scores_per_class)}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to compute similarity scoresOriginal error: {e}Operation log: {print(log)}")

    return {
        "predicted_class": predicted_class,
        "similarity_scores": norm_sim_scores_per_class,
        'operation_log': print_log(log),
    }

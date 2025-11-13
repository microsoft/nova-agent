from smolagents import tool

from nova.toolkits.lazyslide.toolkit_base import PATCH_FEATURES_KEY, TILES_KEY, TISSUES_KEY
from nova.toolkits.lazyslide.toolkit_captioning import LazySlideCaptioningToolKit


@tool
def visualize_text_prompt_similarity_on_wsi_tool(
    wsi_path: str,
    job_dir: str,
    prompt_term: str,
    load_from_src: bool = True,
    text_encoder: str = "conch",
    tissue_key: str = TISSUES_KEY,
    tile_key: str = TILES_KEY,
    tissue_id: int | None = None,
    patch_features_key: str = PATCH_FEATURES_KEY,
    similarity_metric: str = "cosine",
    apply_softmax: bool = True,
    dpi: int = 200,
    cmap: str = "inferno",
    similarity_key_added: str | None = None,
) -> dict:
    """
    Visualize similarity between text prompt term and the tiles in a WSI. Makes a 2 column plot that shows the WSI on left and heatmap of similarity scores on the right.

    By encoding the tiles of a WSI with a vision-language pretrained encoder (only conch and plip are supported),
    this function allows users to visualize how similar each tile is to user-provided text prompts.

    Notes on using this tool:
        - Load the WSI, optionally from a pre-existing Zarr store.
        - Check that tissue segmentation exists under the specified `tissue_key`.
        - Check that tile extraction exists under the specified `tile_key`.
        - Check that patch features exist under the specified `patch_features_key`.
        - If similarity_key_added is provided then it will be used to load the similarity scores from the WSI's tables. New similarity scores will not be computed.
        - If similarity_key_added is not provided, it will be set to f"{patch_features_key}_text_similarity_custom" and similarity scores will be computed and saved to the WSI's tables under this key.
        - Create a 2-column plot:
            - Left: Tissue image with tiles.
            - Right: Heatmap of similarity scores for the prompt term.
        - Save the figure and CSV with per-tile scores for the term. CSV has columns: tile_id,tissue_id,base_height,base_width,base_level,top_left_x,top_left_y,tile_id,prompt_term

    Prerequisites to run this tool:
        - The WSI should be accessible at the provided `wsi_path`.
        - Tissue segmentation must exist for the provided `tissue_key`.
        - Tile extraction must exist for the provided `tile_key`.
        - The environment must have permissions (usually HuggingFace token) to download the specified text encoder model.

    The tool returns:
        dict: Dictionary with:
            - 'similarity_map_save_path': absolute path to the saved similarity figure.
            - 'similarity_scores_csv_save_path': absolute path to the CSV with per-tile scores.
            - 'operations_log': Operations performed during tool call.

    Args:
        wsi_path: Path to the WSI file.
        job_dir: Directory for job outputs and visualization artifacts.
        prompt_term: Text prompt to visualize as a concept.
        load_from_src: Load from existing zarr store if True. Defaults to False.
        text_encoder: Encoder for text embedding. Must be 'conch' or 'plip'.
        tissue_key: Key for tissue segmentation. Defaults to "tissues".
        tile_key: Key for tile extraction. Defaults to "tissue_tiles".
        tissue_id: If set, only visualize for this tissue region.
        patch_features_key: Key for patch features. Must match encoder. Defaults to "uni_tissue_tiles_1024".
        similarity_metric: Similarity metric to use ('cosine' or 'dot'). Defaults to 'cosine'.
        apply_softmax: Whether to apply softmax to similarity scores (default True).
        dpi: DPI for the saved visualization figure.
        cmap: Matplotlib colormap for visualization. Defaults to 'inferno'.
        similarity_key_added: Optional key for pre-computed similarity scores. similairty scores loaded with this key
    """
    cap_tool_kit = LazySlideCaptioningToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=load_from_src)
    return cap_tool_kit.visualize_text_prompt_similarity_on_wsi(
        prompt_term=prompt_term,
        text_encoder=text_encoder,
        tissue_key=tissue_key,
        tile_key=tile_key,
        tissue_id=tissue_id,
        patch_features_key=patch_features_key,
        similarity_metric=similarity_metric,
        apply_softmax=apply_softmax,
        dpi=dpi,
        cmap=cmap,
        similarity_key_added=similarity_key_added,
    )


@tool
def predict_wsi_label_tool(
    wsi_path: str,
    job_dir: str,
    classes: list[str],
    text_encoder: str = "conch",
    tissue_key: str = TISSUES_KEY,
    tile_key: str = TILES_KEY,
    patch_features_key: str = PATCH_FEATURES_KEY,
    method: str = "titan",
) -> dict:
    """
    Compute the probability of the WSI belonging to each of the specified classes.

    `titan` and `mi-zero` are supported. `titan` is significantly preferred over `mi-zero` by the community as it is based on a strong foundation model.
    For method 'titan', a state-of-the-art slide encoder is used to acquire probabilities for classes.
    For method 'mi-zero', patch-level features are compared to text embeddings, and the similarity scores are averaged across patches.

    Requirements for `method` options:
        - Tissue must be segmented using extract_tissue_in_wsi_tool and stored using `tissue_key`
        - Tissue tiles must be extracted using extract_tissue_tiles_in_wsi_tool and stored using `tile_key`
        - Patch feature extraction must be done before using the tool and results should exist in `patch_features_key`
        - To use `titan` method to predict labels:
            - `conch_v1.5` patch embeddings should exist in the object stored by `patch_features_key`.
        - To use `mi-zero` method to predict labels:
            - `conch` or `plip` patch embeddings should exist in the object stored by `patch_features_key`.

    Notes on using this tool:
        - Load the WSI from a pre-existing Zarr store. If no Zarr store exists, then raises RunTimeError
        - Check that tissue segmentation exists under the specified `tissue_key`.
        - Check that tile extraction exists under the specified `tile_key`.
        - Check that patch features exist under the specified `patch_features_key`.
        - Use the specified method to get probabilities per class defined in `classes`
        - Only provide text_encoder when mi-zero method is used

    Prerequisites to run this tool:
        - The WSI should be accessible at the provided `wsi_path`.
        - Tissue segmentation must exist for the provided `tissue_key`.
        - Tile extraction must exist for the provided `tile_key`.
        - Patch features must exist for the provided `patch_features_key`.

    The tool returns:
        dict: Dictionary with the following key:
            - 'probs_for_classes': Dictionary mapping class names to predicted probabilities.
            - 'operations_log': Operations performed during tool call.

    Args:
        wsi_path: Path to the WSI file.
        job_dir: Output/job directory.
        classes: List of class labels/prompts to predict.
        text_encoder: Text encoder for embeddings (default: "conch"). Only "conch" and "plip" are supported. ONLY provide when method is mi-zero
        tissue_key: Key for tissue segmentation. Default: "tissues".
        tile_key: Key for tiles. Default: "tissue_tiles".
        patch_features_key: Feature key for patch features. Default: "uni_tissue_tiles_1024".
        method: Inference method. Must be `titan` or `mi-zero`.
    """
    cap_tool_kit = LazySlideCaptioningToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=True)
    return cap_tool_kit.predict_wsi_label(
        classes=classes,
        text_encoder=text_encoder,
        tissue_key=tissue_key,
        tile_key=tile_key,
        patch_features_key=patch_features_key,
        method=method,
    )


@tool
def generate_wsi_report_with_prism_tool(
    wsi_path: str,
    job_dir: str,
    prompt: list[str],
    load_from_src: bool = True,
    tissue_key: str = TISSUES_KEY,
    tile_key: str = TILES_KEY,
    patch_features_key: str = PATCH_FEATURES_KEY,
) -> dict:
    """
    Generate a Prism-based pathology report for a WSI. Currently this tool is not supported due to issues with the Prism model's requirement of specific trasnformers library.

    Notes on using this tool:
        - Load the WSI, optionally from a pre-existing Zarr store.
        - Check that tissue segmentation exists under the specified `tissue_key`.
        - Check that tile extraction exists under the specified `tile_key`.
        - Check that patch features exist under the specified `patch_features_key`.
        - Use the Prism model to generate a caption/report based on the provided prompt.

    Prerequisites to run this tool:
        - The WSI should be accessible at the provided `wsi_path`.
        - Tissue segmentation must exist for the provided `tissue_key`.
        - Tile extraction must exist for the provided `tile_key`.
        - Patch features must exist for the provided `patch_features_key`.

    The tool returns:
        dict: A dictionary containing:
            - 'slide_caption': The generated caption or report for the WSI.
            - 'operations_log': Operations performed during tool call.

    Args:
        wsi_path: Path to the WSI file.
        job_dir: Directory for intermediate and output files.
        prompt: List of user prompt strings to guide caption generation.
        load_from_src: Whether to load WSI from an existing zarr store. Defaults to False.
        tissue_key: Key for tissue segmentation results in the WSI object. Defaults to "tissues".
        tile_key: Key for tile extraction results in the WSI object. Defaults to "tissue_tiles".
        patch_features_key: Key for patch feature extraction results. Defaults to "uni_tissue_tiles_1024".
    """
    cap_tool_kit = LazySlideCaptioningToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=load_from_src)
    return cap_tool_kit.generate_wsi_report_with_prism(
        prompt=prompt,
        tissue_key=tissue_key,
        tile_key=tile_key,
        patch_features_key=patch_features_key,
    )


@tool
def caption_single_wsi_tool(
    wsi_path: str,
    job_dir: str,
    clustering_key: str,
    load_from_src: bool = True,
    tissue_key: str = TISSUES_KEY,
    tile_key: str = TILES_KEY,
    patch_features_key: str = PATCH_FEATURES_KEY,
    user_instructions: str | None = None,
    caption_model: str = "gpt-4.1",
    summary_model: str = "gpt-4.1",
    summary_temperature: float = 1.0,
    n_tiles_to_select: int = 100,
) -> dict:
    """
    Generate a caption for a single WSI by first clustering patch features, then captioning each cluster, and then summarizing the captions.

    Optionally, the user can provide instructions to guide the caption generation process. The user_instructions will be added to the prompt to the summary generation prompt.

    Notes on using this tool:
        - Load the WSI, optionally from a pre-existing Zarr store.
        - Check that tissue segmentation exists under the specified `tissue_key`.
        - Check that tile extraction exists under the specified `tile_key`.
        - Check that patch features exist under the specified `patch_features_key`.
        - Check for clustering results under the specified `clustering_key`.
        - For each cluster, select n_tiles_to_select tiles randomly and generate a caption using the specified caption model. Empty clusters will be skipped.
        - Generate a summary of the captions using the specified summary model and temperature.

    Prerequisites to run this tool:
        - The WSI should be accessible at the provided `wsi_path`.
        - Tissue segmentation must exist for the provided `tissue_key`.
        - Tile extraction must exist for the provided `tile_key`.
        - Patch features must exist for the provided `patch_features_key`.
        - Clustering results must exist under the specified `clustering_key`.
        - The environment must have permissions (usually HuggingFace token) to download the specified caption and summary models.

    The tool returns:
        dict: A dictionary containing:
            - 'report': The generated caption for the WSI.
            - 'operations_log': Operations performed during tool call.

    Args:
        wsi_path: Path to the WSI file.
        job_dir: Directory for intermediate and output files.
        clustering_key: Key for clustering results in the WSI object.
        load_from_src: Whether to load WSI from an existing zarr store. Defaults to True.
        tissue_key: Key for tissue segmentation results in the WSI object. Defaults to "tissues".
        tile_key: Key for tile extraction results in the WSI object. Defaults to "tissue_tiles".
        patch_features_key: Key for patch feature extraction results. Defaults to "uni_tissue_tiles_1024".
        user_instructions: Optional user instructions to guide caption generation.
        caption_model: Model to use for generating captions. Defaults to "gpt-4.1".
        summary_model: Model to use for generating the summary of captions. Defaults to "gpt-4.1".
        summary_temperature: Temperature setting for the summary model. Defaults to 1.0.
        n_tiles_to_select: Number of tiles to select from each cluster for captioning. Defaults to 10.
    """

    cap_tool_kit = LazySlideCaptioningToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=load_from_src)
    return cap_tool_kit.caption_single_wsi(
        clustering_key=clustering_key,
        tissue_key=tissue_key,
        tile_key=tile_key,
        patch_features_key=patch_features_key,
        user_instructions=user_instructions,
        caption_model=caption_model,
        summary_model=summary_model,
        summary_temperature=summary_temperature,
        n_tiles_to_select=n_tiles_to_select,
    )


@tool
def score_tiles_by_text_in_a_wsi_tool(
    wsi_path: str,
    job_dir: str,
    classes: list[str],
    load_from_src: bool = True,
    tissue_key: str = TISSUES_KEY,
    tile_key: str = TILES_KEY,
    patch_features_key: str = PATCH_FEATURES_KEY,
    text_encoder: str = "conch",
    similarity_metric: str = "cosine",
    apply_softmax: bool = False,
    similarity_key_to_add: str | None = None,
) -> dict:
    """
    Given prompts in the classes list, the tool will score all of the tiles in a WSI based on the similarity of the tile features to the text embeddings of the prompts.
    This tool is not meant to score individual images but rather to score all tiles in a WSI.

    Notes on using this tool:
        - Load the WSI, optionally from a pre-existing Zarr store.
        - Check that tissue segmentation exists under the specified `tissue_key`.
        - Check that tile extraction exists under the specified `tile_key`.
        - Check that patch features exist under the specified `patch_features_key`. Patch features must be extracted with `conch` or `plip` text encoder.
        - Compute text embeddings for the provided class labels using the specified text encoder.
        - Compute similarity scores between patch features and text embeddings.
        - Apply softmax to similarity scores if specified.
        - Save the similarity scores to the WSI's tables under the key `similarity_key_to_add` or `{patch_features_key}_text_similarity_custom` by default.
        - Save the similarity scores to a CSV file in the job directory with columns: tile_id,tissue_id,base_height,base_width,base_level,top_left_x,top_left_y,similarity with prompt term 1, similarity with prompt term 2, etc.

    Prerequisites to run this tool:
        - The WSI should be accessible at the provided `wsi_path`.
        - Tissue segmentation must exist for the provided `tissue_key`.
        - Tile extraction must exist for the provided `tile_key`.
        - Patch features must exist for the provided `patch_features_key`.
        - The environment must have permissions (usually HuggingFace token) to download the specified text encoder model.

    The tool returns:
        dict: Dictionary with:
            - 'similarity_scores_csv_save_path': absolute path to the CSV with per-tile scores.
            - 'similarity_key_to_add': Key under which similarity scores were saved in the WSI's tables.
            - 'operations_log': Operations performed during tool call.

    Args:
        wsi_path: Path to the WSI file.
        job_dir: Directory for intermediate and output files.
        classes: list of individual prompts to score the tiles by
        load_from_src: load from existing zarr store if True. Defaults to True.
        tissue_key: Key for tissue segmentation. Defaults to "tissues".
        tile_key: Key for tile extraction. Defaults to "tiles".
        patch_features_key: Key for patch features. Defaults to "patch_features".
        text_encoder: Text encoder to use for computing embeddings. Defaults to "conch".
        similarity_metric: Similarity metric to use for scoring. Defaults to "cosine".
        apply_softmax: Whether to apply softmax to similarity scores. Defaults to False.
        similarity_key_to_add: Key under which to add similarity scores in the WSI's tables. Defaults to None. If None, it will be set to `{patch_features_key}_text_similarity_custom`.
            If the key already exists, the tool will load the similarity scores from the WSI's tables.

    """

    cap_tool_kit = LazySlideCaptioningToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=load_from_src)
    return cap_tool_kit.score_tiles_in_wsi(
        classes=classes,
        tissue_key=tissue_key,
        tile_key=tile_key,
        patch_features_key=patch_features_key,
        text_encoder=text_encoder,
        similarity_metric=similarity_metric,
        apply_softmax=apply_softmax,
        similarity_key_to_add=similarity_key_to_add,
    )

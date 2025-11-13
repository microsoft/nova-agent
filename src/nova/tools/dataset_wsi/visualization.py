from smolagents import tool

from nova.toolkits.trident.toolkit_dataset_heatmaps import TridentVisualizationToolkit


@tool
def dataset_of_wsi_create_score_heatmap_tool(
    job_dir: str,
    wsi_source: str,
    output_dir: str,
    scores_path: str,
    target_magnification: int,
    patch_size: int,
    overlap: int,
    scores_key: str = 'attention_scores',
    normalize: bool = True,
    num_top_patches_to_save: int = 10,
    cmap: str = 'coolwarm',
    search_nested: bool = False,
    saveto_folder: str | None = None,
    skip_specific_wsi: list[str] | None = None,
    keep_only_these_wsi: list[str] | None = None,
    max_workers: int = 10,
) -> dict:
    """
    Overlay and visualize patch-level scores on a dataset of WSIs, generating color heatmaps and extracting top-k high-scoring patches per slide.

    For each WSI with available scores, this tool generates:
        - A heatmap overlay image of the scores on the slide.
        - The top-k highest scoring patches as image tiles.

    Scores may be attention weights, patch-level similarities, or other numeric values.

    Notes on using this tool:
        - Loads patch-level scores from `.h5` files in `scores_path`, each named `{slide_name}.h5` with a dataset `scores_key`.
        - Loads patch coordinates from `{job_dir}/{saveto_folder}/patches/{slide_name}_patches.h5`.
            Each coordinate file must include:
              - `coords`: dataset of patch coordinates.
              - `patch_size_level0`: attribute with patch size at level 0.
        - Locates the original WSI file from `wsi_source` (matched by slide name).
        - Outputs are organized under `{output_dir}/{slide_name}/` for each WSI:
            - Heatmap image.
            - Directory with top-k patch crops.
        - Skips WSIs listed in `skip_specific_wsi` or those missing score files.

    Prerequisites to run this tool:
        - Each WSI must have:
            - A score `.h5` file in `scores_path` with dataset `scores_key`.
            - A patch coordinates `.h5` file in `{job_dir}/{saveto_folder}/patches/`.
            - The corresponding raw WSI file in `wsi_source`.
        - If `saveto_folder` is not provided, defaults to `{target_magnification}x_{patch_size}px_{overlap}px_overlap`.

    The tool returns:
        dict: Dictionary with keys:
            - `heatmaps_save_paths` : dictionary with keys:
                - `slide_id1` : dictionary with keys:
                    - `heatmap` : absolute path to the PNG heatmap
                    - `topk_patches_dir` : absolute path to the directory containing top-k patch crops (saved at level 0 magnification of corresponding WSI). FIle name format: top_{idx}_score_{attention_score}.png
                                            idx goes from 0 to {num_top_patches_to_save - 1}. Higher idx means higher score!
    Args:
        job_dir: Path to the job directory containing folder for patches file.
        wsi_source: Path to the directory containing WSI image files.
        output_dir: Path where heatmaps and top-k patches will be saved.
        scores_path: Directory containing `.h5` score files named `{slide_name}.h5`.
        target_magnification: Magnification level used when patches were extracted (e.g., 20).
        patch_size: Size of each patch in pixels (e.g., 512).
        overlap: Overlap between adjacent patches in pixels (>=0).
        scores_key: Optional. Dataset name inside `.h5` score files Default: "attention_scores".
        normalize: Optional. If True, normalize scores before visualization Default: True.
        num_top_patches_to_save: Optional. Number of highest-scoring patches to save per slide. Must be >=1. Default: 10.
        cmap: Optional. Colormap for heatmap visualization Default: "coolwarm".
        search_nested: Optional. If True, search `wsi_source` recursively for WSIs Default: False.
        saveto_folder: Optional. Subfolder under `job_dir` where patch coordinate files are found.
                       If None, defaults to `{target_magnification}x_{patch_size}px_{overlap}px_overlap`. Default: None.
        skip_specific_wsi: Optional. List of WSI names (without extension) to exclude Default: None.
        keep_only_these_wsi: Optional. List of WSI names (without extension) to keep.
                             If None, all WSIs are processed after filtering out skipped ones. Default: None
        max_workers: Optional. Maximum workers to use for data loading. Default: 10.
    """

    tridentoolkit_instance = TridentVisualizationToolkit(
        job_dir=job_dir,
        wsi_source=wsi_source,
        search_nested=search_nested,
        max_workers=max_workers,
        skip_specific_wsi=skip_specific_wsi,
        keep_only_these_wsi=keep_only_these_wsi,
    )

    asset_dict = tridentoolkit_instance.generate_heatmaps(
        scores_dir=scores_path,
        target_magnification=target_magnification,
        patch_size=patch_size,
        overlap=overlap,
        output_dir=output_dir,
        normalize=normalize,
        num_top_patches_to_save=num_top_patches_to_save,
        cmap=cmap,
        saveto_folder=saveto_folder,
        scores_key=scores_key,
    )

    return asset_dict

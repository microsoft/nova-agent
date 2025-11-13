import spatialdata as sd
import zarr
from smolagents import tool
from wsidata import open_wsi

from nova.toolkits.lazyslide.toolkit_base import TILES_KEY, TISSUES_KEY
from nova.toolkits.lazyslide.toolkit_processing import LazySlideProcessingToolKit


@tool
def read_zarr_data_tool(path: str) -> dict[str, object]:
    """
    Reads a Zarr file and returns the parsed data. Use this always to read Zarr files.

    The tool returns:
        dict[str, object]: A dictionary containing:
            - 'data': The parsed Zarr data.
            - 'data_str': A string representation of the Zarr data.

    Args:
        path: The file path to the Zarr file.
    """
    data = sd.read_zarr(path)
    data_str = str(data)
    asset_dict = {
        'data': data,
        'data_str': data_str,
    }

    return asset_dict


@tool
def retrieve_properties_from_wsi_tool(
    wsi_path: str,
    job_dir: str,
    tissue_key: str | None = None,
    tile_key: str | None = None,
    patch_features_key: str | None = None,
) -> dict:
    """
    Retrieve key properties of a single WSI and optionally report tissue counts, tile counts, and patch-feature metadata.

    This tool inspects the WSI pyramid (levels, base dimensions, magnification, MPP) via the lazyslide backend.
    If provided, it also (a) counts tissues for `tissue_key`, (b) counts tiles for `tile_key`, and
    (c) checks whether patch features exist for `patch_features_key` and records the feature dimensionality.
    All gathered properties are written to a JSON file under `job_dir`.

    Notes on using this tool:
        - The tool first loads the WSI and initializes it with a Zarr store if the Zarr store (checks at `{job_dir}/{wsi_name_without_extension}.zarr`).
            If it is not, then a clean blank WSI is initialized.
        - If `tissue_key` is provided and present, includes the number of tissue regions.
        - If `tile_key` is provided and present, includes the number of tiles.
        - If `patch_features_key` is provided and present, includes existence and embedding dimensionality of patch features.

    Directory structure & files created:
        - Writes gathered properties to `{job_dir}/{wsi_name_without_extension}/{wsi_name_without_extension}_properties.json`.

    Prerequisites to run this tool:
        - `wsi_path` must be accessible and readable by the lazyslide backend.
        - `job_dir` must exist and be writable.
        - If `tissue_key`, `tile_key`, or `patch_features_key` are provided, the corresponding assets should exist to report those properties.

    The tool returns:
        dict: Dictionary containing slide and pyramid properties:
            - 'n_levels': int. Number of pyramid levels available in the WSI.
            - 'base_height': int. Height in pixels at level 0.
            - 'base_width': int. Width in pixels at level 0.
            - 'base_magnification': float. Nominal magnification at level 0 (e.g., 40.0).
            - 'base_mpp': float. Microns-per-pixel at level 0 (e.g., 0.2465).
            - 'levels': dict. Maps level names (e.g., 'level_0', 'level_1', ...) to a dict with:
                - 'level': int. Numeric level index (0 to n_levels-1).
                - 'img_height_at_level': int. Image height (pixels) at this level.
                - 'img_width_at_level': int. Image width (pixels) at this level.
                - 'downsample_to_level_wrt_level0': float. Downsample factor relative to level 0.
                - 'magnification_at_level': float. Nominal magnification at this level.
                - 'mpp_at_level': float. Microns-per-pixel at this level.
            - 'wsi_file_path': str. Absolute path to the WSI file.
            - 'wsi_reader_name': str. Name of the WSI reader backend (e.g., lazyslide implementation).
            - 'n_tissues': int. Number of tissue regions (present only if `tissue_key` is provided and exists).
            - 'n_tiles_for_{tile_key}': int. Number of tiles associated with `tile_key` (present only if `tile_key` is provided and exists).
            - '{patch_features_key}_exists': bool. Whether patch features for `patch_features_key` exist (present only if provided).
            - '{patch_features_key}_embedding_dim': int. Feature dimensionality for `patch_features_key` (present only if it exists).
            - 'asset_dict_path': str. Absolute path to the output JSON file with these properties.
            - 'operations_log': str. Formatted log of the operations performed during this tool call.

    Args:
        wsi_path: Path to the input WSI file.
        job_dir: Directory where output files and assets will be stored.
        tile_key: Optional. Tile key for fetching tile counts, if present. Default: None.
        tissue_key: Optional. Tissue key for fetching tissue counts, if present. Default: None.
        patch_features_key: Optional. Key for fetching patch features and their dimensionality, if present. Default: None.
    """

    proc_tool_kit = LazySlideProcessingToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=True)
    return proc_tool_kit.retrieve_properties_from_wsi(
        tile_key=tile_key, tissue_key=tissue_key, patch_features_key=patch_features_key
    )


@tool
def extract_tissue_in_wsi_tool(
    wsi_path: str,
    job_dir: str,
    load_from_src: bool = False,
    overwrite: bool = False,
    tissue_key: str = TISSUES_KEY,
) -> dict:
    """
    Run tissue segmentation on a single WSI and save the results to a Zarr store. Tool can be used for datasets of WSI, but does not have optimizations so should be used only when necessary when working with multiple WSI.

    Tissue segmentation separates tissue regions from the background. This is a prerequisite for visualization,
    tiling, and feature extraction on WSIs. Artifact removal is not supported.

    Notes on using this tool:
        - Loads the WSI either from the raw source file or, if `load_from_src=True`, from an existing Zarr store at
          `{job_dir}/{wsi_name_without_extension}.zarr`.
        - If `load_from_src=False`, any existing Zarr store at `{job_dir}/{wsi_name_without_extension}.zarr` will be
          **erased and replaced with a new blank store** before reloading the WSI.
        - If `overwrite=True`, tissue segmentation is run and saved under the specified `tissue_key`.
        - If `overwrite=False`, the tool checks if segmentation exists under `tissue_key`; if present, it reuses that result.
        - After this tool finishes, you can reload the WSI Zarr store (`wsi_zarr`) and access the tissue segmentation
          results via `wsi_zarr.shapes[tissue_key]`. This object is a GeoDataFrame with one row per independent tissue region.
          The GeoDataFrame contains the following columns:
            - `tissue_id`: Unique integer identifier for each region.
            - `geometry`: Polygon geometry of the tissue region (pixel space, relative to level 0 mpp).
            - `area`, `area_filled`, `convex_area`, `solidity`, `convexity`
            - `axis_major_length`, `axis_minor_length`, `eccentricity`, `orientation`

    Directory structure & files created:
        - `{job_dir}/{wsi_name_without_extension}.zarr/`
          Zarr store containing the WSI pyramid, metadata, and tissue segmentation results.
          Tissue results are stored under `shapes[tissue_key]` inside the Zarr hierarchy.

    Prerequisites to run this tool:
        - `wsi_path` must be a valid WSI readable by the lazyslide backend.
        - `job_dir` must exist and be writable.
        - If `load_from_src=True`, a compatible Zarr store at `{job_dir}/{wsi_name_without_extension}.zarr` should exist.

    The tool returns:
        dict: Dictionary summarizing tissue segmentation.
            - 'wsi_store_path': str. Path to the output Zarr store file containing segmentation.
            - 'tissue_ids': list[int]. List of unique tissue IDs (from 0 to N_unique-1).
            - 'tissue_key': str. Key under which segmentation results are stored (default: "tissues").
            - 'operations_log': str. Log of operations performed during this tool call.

    Args:
        wsi_path: Path to the input WSI file.
        job_dir: Directory where outputs and the Zarr store will be saved.
        load_from_src: Optional. If True, loads the WSI from an existing Zarr store at
                       `{job_dir}/{wsi_name_without_extension}.zarr`. If False, deletes any existing Zarr store
                       at that path and creates a new blank one. Default: False.
        overwrite: Optional. If True, re-runs tissue segmentation even if results exist under `tissue_key`.
                   If False, reuses existing segmentation if available. Default: False.
        tissue_key: Optional. Key under which tissue segmentation results are stored in the Zarr hierarchy.
                    Default: "tissues".
    """

    proc_tool_kit = LazySlideProcessingToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=load_from_src)
    return proc_tool_kit.extract_tissue_in_wsi(tissue_key=tissue_key, overwrite=overwrite)


@tool
def extract_tissue_tiles_in_wsi_tool(
    wsi_path: str,
    job_dir: str,
    load_from_src: bool = True,
    tissue_key: str = TISSUES_KEY,
    tile_key: str = TILES_KEY,
    tile_px: int = 256,
    mpp: float = 0.5,
    overlap: float = 0.0,
    overwrite: bool = False,
) -> dict:
    """
    Extract tile (patch) coordinates from a single WSI at provided path. Tool can be used for datasets of WSI, but does not have optimizations so should be used only when necessary when working with multiple WSI.

    This step takes tissue regions and tessellates them into smaller patches (tiles) defined by `tile_px`, `mpp`,
    and `overlap`. It is necessary for visualizing tiles and for subsequent tile feature extraction. Saving only
    tiles based on a minimum tissue proportion is not supported by this tool.

    Notes on using this tool:
        - Loads the WSI, optionally from a pre-existing Zarr store at `{job_dir}/{wsi_name_without_extension}.zarr`.
        - Requires that tissue segmentation results exist under the specified `tissue_key`.
        - If `overwrite=True`, re-runs tiling and saves under the specified `tile_key`.
        - If `overwrite=False`, checks if tiles already exist under the specified `tile_key` and reuses them.
        - Writes the WSI object with tiles back to the Zarr store.
        - After tool execution, load the Zarr store (`wsi_zarr`) and access tiling results via `wsi_zarr.shapes[tile_key]`, which
          is a GeoDataFrame with one row per tile and the following columns:
            - `tile_id`: Unique identifier for each tile (continues across different tissue_ids without reset).
            - `tissue_id`: Unique identifier of the tissue region that the tile belongs to.
            - `geometry`: Polygon geometry of the tile in pixel space relative to level 0 mpp.

    Directory structure & files created:
        - `{job_dir}/{wsi_name_without_extension}.zarr/`
          Zarr store containing the WSI pyramid, metadata, and tiling results.
          Tile polygons are stored under `shapes[tile_key]` in the Zarr hierarchy.

    Prerequisites to run this tool:
        - `wsi_path` must point to a valid WSI readable by the lazyslide backend.
        - `job_dir` must exist and be writable.
        - Tissue segmentation must already exist in the Zarr store under `tissue_key`.

    The tool returns:
        dict: Dictionary summarizing the patch extraction.
            - 'wsi_store_path': str. Path to the Zarr store containing the WSI and tiling results.
            - 'tissue_key': str. Key under which tissue segmentation is stored in the Zarr hierarchy.
            - 'tile_key': str. Key under which tile extraction results are stored in the Zarr hierarchy.
            - 'num_patches': int. Number of patches (tiles) generated for the slide.
            - 'operations_log': str. Log of operations performed during this tool call.

    Args:
        wsi_path: Path to the input WSI file.
        job_dir: Directory where outputs and Zarr store will be saved.
        load_from_src: Optional. If True, loads the WSI from an existing Zarr store.
                       If False, erases any existing store and creates a new blank one. Default: True.
        tissue_key: Optional. Key under which tissue segmentation results are stored.
                    Default: "tissues".
        tile_key: Optional. Key under which tile extraction results are stored.
                  Typically formatted as `tiles_px{tile_px}_mpp{mpp}_overlap{overlap}`.
                  Default: "tissue_tiles".
        tile_px: Optional. Tile (patch) size in pixels. Default: 256.
        mpp: Optional. Target microns per pixel for tiling. Default: 0.5.
        overlap: Optional. Overlap fraction between adjacent tiles. Default: 0.0.
        overwrite: Optional. If True, re-runs tiling even if results exist. Default: False.
    """

    proc_tool_kit = LazySlideProcessingToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=load_from_src)
    return proc_tool_kit.extract_coordinates_in_wsi(
        tissue_key=tissue_key,
        tile_key=tile_key,
        tile_px=tile_px,
        mpp=mpp,
        overlap=overlap,
        overwrite=overwrite,
    )


@tool
def extract_patch_features_in_wsi_tool(
    wsi_path: str,
    job_dir: str,
    load_from_src: bool = True,
    tissue_key: str = TISSUES_KEY,
    tile_key: str = TILES_KEY,
    patch_features_key: str | None = None,
    model: str = "conch_v1.5",
    batch_size: int = 64,
    num_workers: int = 10,
    overwrite: bool = False,
) -> dict:
    """
    Extract patch-level features for a single WSI at the provided path using the specified patch encoder model. Tool can be used for datasets of WSI, but does not have optimizations so should be used only when necessary when working with multiple WSI.

    Notes on using this tool:
        - Loads the WSI, optionally from a pre-existing Zarr store at `{job_dir}/{wsi_name_without_extension}.zarr`.
        - Requires that tissue segmentation exists under `tissue_key`.
        - Requires that tile extraction exists under `tile_key`.
        - If `overwrite=True`, re-runs feature extraction and saves results under `patch_features_key`.
        - If `overwrite=False`, checks if features already exist under `patch_features_key` and reuses them if present.
        - Writes the WSI object with patch features back to the Zarr store.
        - If `patch_features_key` is None, it defaults to `{model}_{tile_key}` for readability.
        - Supported patch encoder models: ['conch_v1.5', 'conch', 'uni'].
            SUGGESTED preference: 'conch_v1.5' → 'conch' → 'uni' (unless specified otherwise).
        - After execution, patch features can be accessed by loading the Zarr store and using `wsi_zarr[patch_features_key]`,
          which is an AnnData object with attributes:
            - `X`: numpy array of shape (n_patches, embedding_dim) containing extracted patch features.

    Directory structure & files created:
        - `{job_dir}/{wsi_name_without_extension}.zarr/`
          Zarr store containing the WSI pyramid, metadata, and patch feature extraction results.
          Features are stored under `patch_features_key` as an AnnData object.

    Prerequisites to run this tool:
        - `wsi_path` must be a valid WSI readable by the lazyslide backend.
        - `job_dir` must exist and be writable.
        - Tissue segmentation must already exist under `tissue_key`.
        - Tile extraction must already exist under `tile_key`.
        - User must have access rights to download/load the specified model (e.g., HuggingFace token if required).
        - Compute hardware (CPU/GPU) must have sufficient memory/resources to run inference with the given `batch_size`.

    The tool returns:
        dict: Dictionary summarizing patch feature extraction.
            - 'tissue_key': str. Key for tissue segmentation used.
            - 'tile_key': str. Key for tile extraction used.
            - 'patch_features_key': str. Key under which extracted features are stored in the Zarr store.
            - 'embedding_dim': int. Dimensionality of the extracted patch features.
            - 'operations_log': str. Log of operations performed during this tool call.

    Args:
        wsi_path: Path to the input WSI file.
        job_dir: Directory where outputs and Zarr store will be saved.
        load_from_src: Optional. If True, loads the WSI from an existing Zarr store at
                       `{job_dir}/{wsi_name_without_extension}.zarr`. If False, erases any existing store
                       and creates a new blank one. Default: True.
        tissue_key: Optional. Key under which tissue segmentation results are stored. Default: "tissues".
        tile_key: Optional. Key under which tile extraction results are stored.
                  Typically formatted as `tiles_px{tile_px}_mpp{mpp}_overlap{overlap}`. Default: "tissue_tiles".
        patch_features_key: Optional. Key under which extracted features are stored.
                            If None, defaults to `{model}_{tile_key}`. Default: None.
        model: Optional Name of the patch encoder model to use. Default: "conch_v1.5".
        batch_size: Batch size for inference. Default: 64.
        overwrite: If True, re-runs feature extraction even if results already exist. Default: False.
        num_workers: Optional. Number of worker threads to use for data loading and preprocessing. Default: 10.
    """

    proc_tool_kit = LazySlideProcessingToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=load_from_src)
    return proc_tool_kit.extract_patch_features_in_wsi(
        tissue_key=tissue_key,
        tile_key=tile_key,
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        overwrite=overwrite,
        patch_features_key=patch_features_key,
    )


@tool
def encode_wsi_tool(
    wsi_path: str,
    job_dir: str,
    patch_features_key: str,
    slide_encoder: str = "mean",
    slide_encoder_key: str = "mean_slide_repr",
    load_from_src: bool = True,
    tissue_key: str = TISSUES_KEY,
    tile_key: str = TILES_KEY,
    overwrite: bool = False,
) -> dict:
    """
    Encode a single WSI into a slide-level representation using the specified slide encoder. Tool can be used for datasets of WSI, but does not have optimizations so should be used only when necessary when working with multiple WSI.

    Notes on using this tool:
        - Loads the WSI, optionally from a pre-existing Zarr store at `{job_dir}/{wsi_name_without_extension}.zarr`.
        - Requires that tissue segmentation exists under `tissue_key`.
        - Requires that tile extraction exists under `tile_key`.
        - Requires that patch features exist under `patch_features_key`.
        - If `overwrite=True`, re-runs slide encoding and saves results under `slide_encoder_key`.
        - If `overwrite=False`, checks if results already exist under `slide_encoder_key` and reuses them if present.
        - Writes the WSI object with slide-level representation back to the Zarr store.
        - Slide-level representation is stored at `wsi[patch_features_key].varm[slide_encoder_key]`.
        - If `slide_encoder_key` is None, it defaults to `{slide_encoder}_slide_representation`.
        - After execution, you can load the saved WSI Zarr store (`wsi_zarr`) and access the slide encoding at:
          `wsi_zarr[patch_features_key].varm[slide_encoder_key]`, which is a numpy array of shape `(embedding_dim,)`.

    Directory structure & files created:
        - `{job_dir}/{wsi_name_without_extension}.zarr/`
          Zarr store containing the WSI pyramid, metadata, patch features, and slide-level representation.
          The slide-level representation is stored under:
            `{patch_features_key}.varm[{slide_encoder_key}]`.

    Prerequisites to run this tool:
        - `wsi_path` must be a valid WSI readable by the lazyslide backend.
        - `job_dir` must exist and be writable.
        - Tissue segmentation must already exist under `tissue_key`.
        - Tile extraction must already exist under `tile_key`.
        - Patch features must already exist under `patch_features_key`.
        - For `titan` slide encoder: patch features must be extracted with `conch_v1.5` patch encoder.
        - For `prism` slide encoder: patch features must be extracted with `virchow` patch encoder.
        - User must have access rights to download/load the specified model (e.g., HuggingFace token if required).
        - Compute hardware (CPU/GPU) must have sufficient resources for the model inference.
        - Uses the lazyslide backend to interact with the WSI.

    The tool returns:
        dict: Dictionary summarizing slide encoding.
            - 'tissue_key': str. Key for tissue segmentation used.
            - 'tile_key': str. Key for tile extraction used.
            - 'patch_features_key': str. Key under which extracted patch features are stored.
            - 'slide_encoder_key': str. Key under which slide-level representation is stored.
            - 'embedding_dim': int. Dimensionality of the slide-level representation.
            - 'operations_log': str. Log of operations performed during this tool call.

    Args:
        wsi_path: Path to the input WSI file.
        job_dir: Directory where outputs and the Zarr store will be saved.
        patch_features_key: Key under which extracted patch features are stored.
        slide_encoder: Name of the slide encoder to use (e.g., "mean", "titan", "prism"). Default: "mean".
        slide_encoder_key: Key under which slide-level representation is stored.
                          If None, defaults to "{slide_encoder}_slide_representation". Default: "mean_slide_repr".
        load_from_src: Optional. If True, loads the WSI from an existing Zarr store at
                       `{job_dir}/{wsi_name_without_extension}.zarr`. If False, erases any existing store
                       and creates a new blank one. Default: True.
        tissue_key: Optional. Key under which tissue segmentation results are stored. Default: "tissues".
        tile_key: Optional. Key under which tile extraction results are stored.
                  Typically formatted as `tiles_px{tile_px}_mpp{mpp}_overlap{overlap}`. Default: "tissue_tiles".
        overwrite: If True, re-runs slide encoding even if results exist. Default: False.
        device: Device on which to run inference (e.g., "cuda", "cpu").
                If None, the lazyslide backend chooses automatically. Default: None.
    """

    proc_tool_kit = LazySlideProcessingToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=load_from_src)
    return proc_tool_kit.encode_single_wsi(
        slide_encoder=slide_encoder,
        tissue_key=tissue_key,
        tile_key=tile_key,
        patch_features_key=patch_features_key,
        slide_features_key=slide_encoder_key,
        overwrite=overwrite,
    )


@tool
def check_tissue_segmentation_key_in_wsi_tool(
    wsi_path: str,
    job_dir: str,
    tissue_key: str,
) -> bool:
    """
    Check whether a tissue segmentation exists under the specified `tissue_key` in a WSI Zarr store.

    Notes on using this tool:
        - Always loads the WSI from an existing Zarr store at `{job_dir}/{wsi_name_without_extension}.zarr`.
        - Raises a `RuntimeError` if the Zarr store does not exist.
        - Raises a `KeyError` if `tissue_key` is not found in `wsi.shapes`.

    Directory structure & files used:
        - `{job_dir}/{wsi_name_without_extension}.zarr/`
          Zarr store that must already exist

    Prerequisites to run this tool:
        - The WSI file at `wsi_path` must be readable by the lazyslide backend.
        - A corresponding Zarr store at `{job_dir}/{wsi_name_without_extension}.zarr` must already exist.
        - Tissue segmentation should have been extracted previously and stored under the provided `tissue_key`.

    The tool returns:
        bool:
            - True if the `tissue_key` exists in `wsi.shapes`.
            - Raises `KeyError` if the Zarr store does not exist or if the key is missing.

    Args:
        wsi_path: Path to the input WSI file.
        job_dir: Directory where the Zarr store is expected to exist.
        tissue_key: Key under which tissue segmentation results should be stored in `wsi.shapes`.
    """

    proc_tool_kit = LazySlideProcessingToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=True)
    log = []

    if proc_tool_kit.wsi_store_path.exists():
        wsi = open_wsi(proc_tool_kit.wsi_path, store=str(proc_tool_kit.wsi_store_path))
        log.append(f"\n Loaded WSI successfully from source {proc_tool_kit.wsi_store_path}.")
    else:
        raise RuntimeError(
            f'Zarr file not found at {proc_tool_kit.wsi_store_path}. This means no processing is done on this WSI!'
        )

    # if key is not found, then error message raise by check_tissue_key_in_WSI
    _ = proc_tool_kit._check_tissue_key(wsi=wsi, tissue_key=tissue_key, log=log)

    return True


@tool
def check_tile_key_in_wsi_tool(
    wsi_path: str,
    job_dir: str,
    tile_key: str,
) -> bool:
    """
    Check whether tile extraction results exists under the specified `tile_key` in a WSI Zarr store.

    Notes on using this tool:
        - Always loads the WSI from an existing Zarr store at `{job_dir}/{wsi_name_without_extension}.zarr`.
        - Raises a `RuntimeError` if the Zarr store does not exist.
        - Raises a `KeyError` if the specified `tile_key` is not found in `wsi.shapes`.
        - The recommended format for `tile_key` is `tiles_px{tile_px}_mpp{mpp}_overlap{overlap}`.

    Directory structure & files used:
        - `{job_dir}/{wsi_name_without_extension}.zarr/`
          Zarr store that must already exist and contain tile extraction results under `shapes[tile_key]`.

    Prerequisites to run this tool:
        - The WSI file at `wsi_path` must be readable by the lazyslide backend.
        - A corresponding Zarr store at `{job_dir}/{wsi_name_without_extension}.zarr` must already exist.
        - Tile extraction should have been previously performed and stored under the provided `tile_key`.

    The tool returns:
        bool:
            - True if the `tile_key` exists in `wsi.shapes`.
            - Raises `RuntimeError` if the Zarr store does not exist.
            - Raises `KeyError` if the `tile_key` is missing in the store.

    Args:
        wsi_path: Path to the input WSI file.
        job_dir: Directory where the Zarr store is expected to exist.
        tile_key: Key under which tile extraction results should be stored in `wsi.shapes`.
    """

    proc_tool_kit = LazySlideProcessingToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=True)
    log = []

    if proc_tool_kit.wsi_store_path.exists():
        wsi = open_wsi(proc_tool_kit.wsi_path, store=str(proc_tool_kit.wsi_store_path))
        log.append(f"\n Loaded WSI successfully from source {proc_tool_kit.wsi_store_path}.")
    else:
        raise RuntimeError(
            f'Zarr file not found at {proc_tool_kit.wsi_store_path}. This means no processing is done on this WSI!'
        )

    # if key is not found, then error message raise by check_tile_key_in_WSI
    _ = proc_tool_kit._check_tile_key(wsi=wsi, tile_key=tile_key, log=log)

    return True


@tool
def check_patch_features_key_in_wsi_tool(
    wsi_path: str,
    job_dir: str,
    patch_features_key: str,
) -> bool:
    """
    Check whether patch features results exist under the specified `patch_features_key` in a WSI Zarr store.

    Notes on using this tool:
        - Always loads the WSI from an existing Zarr store at `{job_dir}/{wsi_name_without_extension}.zarr`.
        - Raises a `RuntimeError` if the Zarr store does not exist.
        - Raises a `KeyError` if the specified `patch_features_key` is not found in `wsi.tables`.
        - By convention, `patch_features_key` defaults to `{model}_{tile_key}` (if not explicitly provided), which is preferred for readability.

    Directory structure & files used:
        - `{job_dir}/{wsi_name_without_extension}.zarr/`
          Zarr store that must already exist and contain patch feature results under `tables[patch_features_key]`.

    Prerequisites to run this tool:
        - The WSI file at `wsi_path` must be readable by the lazyslide backend.
        - A corresponding Zarr store at `{job_dir}/{wsi_name_without_extension}.zarr` must already exist.

    The tool returns:
        bool:
            - True if the `patch_features_key` exists in `wsi.tables`.
            - Raises `RuntimeError` if the Zarr store does not exist.
            - Raises `KeyError` if the `patch_features_key` is missing in the store.

    Args:
        wsi_path: Path to the input WSI file.
        job_dir: Directory where the Zarr store is expected to exist.
        patch_features_key: Key under which patch feature extraction results should be stored in `wsi.tables`.
    """

    proc_tool_kit = LazySlideProcessingToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=True)
    log = []

    if proc_tool_kit.wsi_store_path.exists():
        wsi = open_wsi(proc_tool_kit.wsi_path, store=str(proc_tool_kit.wsi_store_path))
        log.append(f"\n Loaded WSI successfully from source {proc_tool_kit.wsi_store_path}.")
    else:
        raise RuntimeError(
            f'Zarr file not found at {proc_tool_kit.wsi_store_path}. This means no processing is done on this WSI!'
        )

    # if key is not found, then error message raise by check_patch_features_key_in_WSI
    _ = proc_tool_kit._check_patch_features_key(wsi=wsi, patch_features_key=patch_features_key, log=log)

    return True


@tool
def check_slide_features_key_in_wsi_tool(
    wsi_path: str,
    job_dir: str,
    patch_features_key: str,
    slide_features_key: str,
) -> bool:
    """
    Check whether slide-level features exist under the specified `slide_features_key` in a WSI.

    Notes on using this tool:
        - Always loads the WSI from an existing Zarr store at `{job_dir}/{wsi_name_without_extension}.zarr`.
        - Raises a `RuntimeError` if the Zarr store does not exist.
        - Raises a `KeyError` if the specified `slide_features_key` is not found in
          `wsi[patch_features_key].varm`.

    Directory structure & files used:
        - `{job_dir}/{wsi_name_without_extension}.zarr/`
          Zarr store that must already exist and contain patch features under `tables[patch_features_key]`
          and slide-level features under `tables[patch_features_key].varm[slide_features_key]`.

    Prerequisites to run this tool:
        - The WSI file at `wsi_path` must be readable by the lazyslide backend.
        - A corresponding Zarr store at `{job_dir}/{wsi_name_without_extension}.zarr` must already exist.
        - Patch features must have been extracted and stored under `patch_features_key`.

    The tool returns:
        bool:
            - True if the `slide_features_key` exists in `wsi[patch_features_key].varm`.
            - Raises `RuntimeError` if the Zarr store does not exist.
            - Raises `KeyError` if the `slide_features_key` is missing in the varm.

    Args:
        wsi_path: Path to the input WSI file.
        job_dir: Directory where the Zarr store is expected to exist.
        patch_features_key: Key under which patch feature results are stored.
        slide_features_key: Key under which slide-level features should be stored in `wsi[patch_features_key].varm`.
    """

    proc_tool_kit = LazySlideProcessingToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=True)
    log = []

    if proc_tool_kit.wsi_store_path.exists():
        wsi = open_wsi(proc_tool_kit.wsi_path, store=str(proc_tool_kit.wsi_store_path))
        log.append(f"\n Loaded WSI successfully from source {proc_tool_kit.wsi_store_path}.")
    else:
        raise RuntimeError(
            f'Zarr file not found at {proc_tool_kit.wsi_store_path}. This means no processing is done on this WSI!'
        )

    # if key is not found, then error message raise by check_slide_features_key_in_WSI
    _ = proc_tool_kit._check_slide_features_key(
        wsi=wsi, patch_features_key=patch_features_key, slide_features_key=slide_features_key, log=log
    )

    return True


@tool
def check_clustering_key_in_wsi_tool(
    wsi_path: str,
    job_dir: str,
    clustering_key: str,
    patch_features_key: str,
) -> bool:
    """
    Check whether a clustering key exists in the AnnData object associated with the given `patch_features_key`.

    Notes on using this tool:
        - Always loads the WSI from an existing Zarr store at `{job_dir}/{wsi_name_without_extension}.zarr`.
        - Raises a `RuntimeError` if the Zarr store does not exist.
        - Ensures that patch features exist under the specified `patch_features_key` by calling
          `check_patch_features_key_in_wsi_tool`.
        - Raises a `KeyError` if the specified `clustering_key` is not found in
          `wsi[patch_features_key].obs` (a pandas DataFrame).
        - Clustering results are typically stored in `.obs` after a clustering method (e.g., k-means, Leiden)
          is applied to patch-level features.

    Directory structure & files used:
        - `{job_dir}/{wsi_name_without_extension}.zarr/`
          Zarr store that must already exist and contain:
            - Patch features under `tables[patch_features_key]`
            - Clustering results stored as a column (`clustering_key`) in `wsi[patch_features_key].obs`

    Prerequisites to run this tool:
        - The WSI file at `wsi_path` must be readable by the lazyslide backend.
        - A corresponding Zarr store at `{job_dir}/{wsi_name_without_extension}.zarr` must already exist.
        - Patch features must have been extracted under `patch_features_key`.

    The tool returns:
        bool:
            - True if the `clustering_key` exists in `wsi[patch_features_key].obs`.
            - Raises `RuntimeError` if the Zarr store does not exist.
            - Raises `KeyError` if the `clustering_key` is missing in `.obs`.

    Args:
        wsi_path: Path to the input WSI file.
        job_dir: Directory where the Zarr store is expected to exist.
        clustering_key: Key under which clustering results are stored (column name in `.obs`).
        patch_features_key: Key under which patch feature results are stored.
    """

    proc_tool_kit = LazySlideProcessingToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=True)
    log = []

    if proc_tool_kit.wsi_store_path.exists():
        wsi = open_wsi(proc_tool_kit.wsi_path, store=str(proc_tool_kit.wsi_store_path))
        log.append(f"\n Loaded WSI successfully from source {proc_tool_kit.wsi_store_path}.")
    else:
        raise RuntimeError(
            f'Zarr file not found at {proc_tool_kit.wsi_store_path}. This means no processing is done on this WSI!'
        )

    # if key is not found, then error message raise by check_clustering_key_in_WSI
    log = proc_tool_kit._check_patch_features_key(wsi=wsi, patch_features_key=patch_features_key, log=log)
    adata = wsi[patch_features_key]
    _ = proc_tool_kit._check_clustering_key(clustering_key=clustering_key, log=log, adata=adata)  # type: ignore[call-arg]

    return True


@tool
def check_reduction_key_in_wsi_tool(
    wsi_path: str,
    job_dir: str,
    reduction_key: str,
    patch_features_key: str,
) -> bool:
    """
    Check if a dimensionality reduction result exists under the specified key in the WSI.

    Notes on using this tool:
        - Loads the WSI from its Zarr store.
        - Ensures that patch features exist under the specified `patch_features_key`.
        - Checks if the `reduction_key` exists in `adata.obsm` of the AnnData object associated with `patch_features_key`.
        - Raises KeyError if the reduction result is not found, or if WSI loading fails.
        - Common reduction keys may include `"umap"`, `"tsne"`, or `"pca"` depending on preprocessing steps.
        - Uses the `lazyslide` backend to interact with the WSI.

    Directory structure & files used:
        - `{job_dir}/{slide_id}.zarr/`
          The Zarr store for the WSI must exist and contain:
            - `tables/{patch_features_key}`: AnnData object holding patch-level features.
              - `.obsm`: container that should include the `reduction_key`.

    Prerequisites to run this tool:
        - The WSI should be accessible at the provided `wsi_path`.
        - Patch features must exist under the provided `patch_features_key`.

    The tool returns:
        bool:
            - True if `reduction_key` exists in `wsi[patch_features_key].obsm`.
            - Raises KeyError otherwise.

    Args:
        wsi_path: Path to the input WSI file.
        job_dir: Directory where outputs and Zarr store are saved.
        reduction_key: Key under which dimensionality reduction results are stored in `.obsm`.
        patch_features_key: Key under which patch features results are stored.
    """

    proc_tool_kit = LazySlideProcessingToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=True)
    log = []

    if proc_tool_kit.wsi_store_path.exists():
        wsi = open_wsi(proc_tool_kit.wsi_path, store=str(proc_tool_kit.wsi_store_path))
        log.append(f"\n Loaded WSI successfully from source {proc_tool_kit.wsi_store_path}.")
    else:
        raise RuntimeError(
            f'Zarr file not found at {proc_tool_kit.wsi_store_path}. This means no processing is done on this WSI!'
        )
    # if key is not found, then error message raise by check_clustering_key_in_WSI
    log = proc_tool_kit._check_patch_features_key(wsi=wsi, patch_features_key=patch_features_key, log=log)
    adata = wsi[patch_features_key]
    _ = proc_tool_kit._check_reduction_key(reduction_key=reduction_key, log=log, adata=adata)  # type: ignore[call-arg]

    return True


@tool
def access_zarr_hierarchy(
    zarr_store_path: str,
) -> dict:
    """
    Inspect the hierarchy of a Zarr store containing a WSI and its associated data.

    Notes on using this tool:
        - Opens the Zarr store in read-only mode using the `zarr` library.
        - Provides a summary of the hierarchy without loading large image/feature data.
        - To access the actual WSI data (e.g., tissue, tiles, patch features), use the `spatialdata` library instead.
        - This tool is only for inspection and debugging of the Zarr file structure.

    Directory structure & files used:
        - `{zarr_store_path}/`
          The Zarr store directory must already exist. It may contain subgroups for:
            - `"shapes"`: tissue and tile segmentation results (GeoDataFrames).
            - `"tables"`: patch features (AnnData objects).
            - Metadata attributes at the root level.

    Prerequisites to run this tool:
        - The path provided to `zarr_store_path` must be a valid Zarr store created by prior processing steps.
        - The store must be accessible in read mode.

    The tool returns:
        dict: Dictionary summarizing the Zarr store hierarchy with keys:
            - 'zarr_store_path' (str): Path to the Zarr store inspected.
            - 'tree' (str): String representation of the hierarchical structure.
            - 'attrs' (list[str]): List of root-level attributes stored in the Zarr.
            - 'groups' (list[str]): List of group names (top-level keys) within the Zarr.

    Args:
        zarr_store_path: Path to the Zarr store whose hierarchy you want to inspect.
    """

    store = zarr.open(zarr_store_path, mode="r")

    # structure of zarr store
    tree = str(store.tree())  # type: ignore[call-arg]

    # attrs
    attrs = [a for a in store.attrs]

    # groups
    groups = [k for k in store.keys()]  # type: ignore[call-arg]

    asset_dict = {
        'zarr_store_path': zarr_store_path,
        'tree': tree,
        'attrs': attrs,
        'groups': groups,
    }

    return asset_dict

from pathlib import Path

import h5py
import numpy as np
from smolagents import tool
from trident.Converter import OPENSLIDE_EXTENSIONS, PIL_EXTENSIONS


@tool
def dataset_of_wsi_get_valid_slide_paths_tool(
    wsi_source: str,
    search_nested: bool = False,
) -> list[str]:
    """
    Find all valid WSIs in a given directory and return absolute paths to all valid WSIs.

    Scans the source directory (optionally recursively) for files with extensions recognized by OpenSlide and PIL,
    returning a sorted list of absolute file paths for valid slides [valid extensions: '.bif', '.czi', '.dcm', '.jpeg',
    '.jpg', '.mrxs', '.ndpi', '.png', '.scn', '.svs', '.svslide', '.tif', '.tiff', '.vms', '.vmu'].

    Prerequisites to run this tool:
        - Directory must exist, otherwise tool will raise a FileNotFoundError

    The tool returns:
        list[str]: Sorted list of absolute paths to valid WSI files.

    Args:
        wsi_source: Path to the directory to search for WSI files.
        search_nested: Optional Whether to search subdirectories recursively (True) or only the top-level (False). Default: False
    """

    wsi_source_obj = Path(wsi_source)

    if not wsi_source_obj.exists():
        raise FileNotFoundError(f"WSI source directory '{wsi_source}' does not exist.")

    allowed_exts = set([e.lower() for e in OPENSLIDE_EXTENSIONS.union(PIL_EXTENSIONS)])
    if search_nested:
        files = wsi_source_obj.rglob('*')
    else:
        files = wsi_source_obj.glob('*')

    valid_paths = []
    for f in files:
        if f.is_file():
            ext = f.suffix.lower()
            if ext in allowed_exts:
                valid_paths.append(str(f.resolve()))

    return sorted(valid_paths)


@tool
def dataset_of_wsi_check_tissue_segmentation_exists_tool(
    job_dir: str,
    wsis: list[str],
    skip_wsi_to_check: list[str] | None = None,
) -> tuple[dict[str, dict], list[str]]:
    """
    Check for the existence of tissue segmentation results for a list of WSIs, with support for skipping specific WSIs.

    For each WSI name (`wsi_name`) in `wsis` list, this tool checks whether the following files exist in the expected directories:
      - GeoJSON tissue contours:      `{job_dir}/contours_geojson/{wsi_name}.geojson`
      - Thumbnail image:              `{job_dir}/thumbnails/{wsi_name}.jpg`
      - Contour visualization image:  `{job_dir}/contours/{wsi_name}.jpg`

    Notes on using this tool:
        - Checks for the presence of segmentation GeoJSON, thumbnail, and contour image for each WSI.
        - Skips filenames without extension listed in `skip_wsi_to_check` and reports them as skipped.

    Prerequisites to run this tool:
        - The tool only checks for existence within `job_dir`
        - The skip list should contain WSI filenames without extensions.

    The tool returns:
        dict: Mapping of WSI name (without extension) to a dict containing:
            - geojson_path: Absolute path to the GeoJSON file [will not exist if geojson_exists=False].
            - geojson_exists: True if GeoJSON exists, False otherwise.
            - thumbnail_path: Absolute path to the thumbnail image [will not exist if thumbnail_exists=False].
            - thumbnail_exists: True if thumbnail exists, False otherwise.
            - contour_path: Absolute path to the contour image [will not exist if contour_exists=False].
            - contour_exists: True if contour image exists, False otherwise.
        list of str: List of absolute paths to skipped files (one entry for each expected file per skipped WSI).

    Args:
        job_dir: Path to the root directory containing segmentation outputs.
        wsis: List of WSI names (without file extensions).
        skip_wsi_to_check: Optional List of WSI names (without extension) to skip for checking. Default: None (all items in `wsis` checked).
    """
    if skip_wsi_to_check is None:
        skip_wsi_to_check = []

    assert len(wsis) > 0, "wsis cannot be empty. Must provide WSI names without extensions to check."

    geojson_dir = Path(job_dir) / 'contours_geojson'
    thumbnail_dir = Path(job_dir) / 'thumbnails'
    contour_dir = Path(job_dir) / 'contours'
    results = {}
    skipped = []

    for wsi_name in wsis:
        geojson_path = geojson_dir / f"{wsi_name}.geojson"
        thumbnail_path = thumbnail_dir / f"{wsi_name}.jpg"
        contour_path = contour_dir / f"{wsi_name}.jpg"

        if wsi_name in skip_wsi_to_check:
            skipped.extend(
                [
                    str(geojson_path.resolve()),
                    str(thumbnail_path.resolve()),
                    str(contour_path.resolve()),
                ]
            )
            continue

        result = {
            "geojson_path": str(geojson_path.resolve()),
            "geojson_exists": geojson_path.exists(),
            "thumbnail_path": str(thumbnail_path.resolve()),
            "thumbnail_exists": thumbnail_path.exists(),
            "contour_path": str(contour_path.resolve()),
            "contour_exists": contour_path.exists(),
        }
        results[wsi_name] = result

    return results, skipped


@tool
def dataset_of_wsi_check_patch_coordinates_exist_and_schema_tool(
    job_dir: str,
    wsis: list[str],
    saveto_folder: str,
    skip_wsi_to_check: list[str] | None = None,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """
    Check for the existence and schema of patch coordinate H5 files for a list of WSIs, with support for skipping selected WSIs.

    For each WSI name in `wsis`, this tool checks whether a corresponding patch coordinate file (`job_dir/{saveto_folder}/patches/{wsi_name}_patches.h5`)
    exists in the expected directory and validates that it contains a 'coords' dataset with 2D shape (N, 2), where N is the number of patches.

    Notes on using this tool:
        - The tool checks for existence of patches file for each WSI and that each file contains a 2D 'coords' dataset of shape
            (N, 2).
        - Skips filenames without extension listed in `skip_wsi_to_check` and reports them as skipped.

    Prerequisites to run this tool:
        - All files are expected in `job_dir/{saveto_folder}/patches/` with the name `{wsi_name}_patches.h5`.
        - The skip list should contain WSI filenames without extensions.

    The tool returns:
        dict: Mapping of WSI name (without extension) to a dict containing:
            - path: Absolute path to the patch coordinate file. [will not exist if path_exists=False]
            - path_exists: True if the file exists, False otherwise.
            - file_schema_ok: True if the file contains a 2D 'coords' dataset (N, 2), False otherwise.
        list of str: List of absolute paths to skipped files (not WSI patch coordinate files).

    Args:
        job_dir: Path to the root directory for patch coordinate files. The job_dir must contain `saveto_folder`.
        wsis: List of WSI names (without file extensions).
        saveto_folder: Folder name under `job_dir` where patch coordinate files are stored (e.g., '20x_512px_0px_overlap').
        skip_wsi_to_check: Optional List of WSI names (without extension) to skip for checking. Default: None (all items in `wsis` checked).
    """
    if skip_wsi_to_check is None:
        skip_wsi_to_check = []

    assert len(wsis) > 0, "wsis cannot be empty. Must provide WSI names without extensions to check."

    dir_with_patches = Path(job_dir) / saveto_folder / 'patches'
    results = {}
    skipped = []

    for wsi_name in wsis:
        if wsi_name in skip_wsi_to_check:
            skipped.append(str(dir_with_patches / f"{wsi_name}_patches.h5"))
            continue

        file_path = dir_with_patches / f"{wsi_name}_patches.h5"
        result = {
            "path": str(file_path.resolve()),
            "path_exists": file_path.exists(),
            "file_schema_ok": False,
        }

        if result["path_exists"]:
            try:
                with h5py.File(file_path, 'r') as f:
                    if "coords" in f:
                        ds = np.array(f["coords"])
                        if hasattr(ds, "shape") and len(ds.shape) == 2 and ds.shape[1] == 2:
                            result["file_schema_ok"] = True
            except Exception:
                result["file_schema_ok"] = False

        results[wsi_name] = result

    return results, skipped


@tool
def dataset_of_wsi_check_patch_features_exist_and_schema_tool(
    job_dir: str,
    wsis: list[str],
    target_magnification: int = 20,
    patch_size: int = 512,
    overlap: int = 0,
    saveto_folder: str | None = None,
    patch_encoder_name: str = "uni_v1",
    skip_wsi_to_check: list[str] | None = None,
    patch_feats_save_folder: str | None = None,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """
    Check for the existence and schema of patch feature H5 files for a list of WSIs, with support for skipping selected WSIs.

    For each WSI name in `wsis`, this tool checks whether a corresponding patch feature file (`{job_dir}/{saveto_folder}/{patch_feats_save_folder}/{wsi_name}.h5`) exists in
    the expected directory and validates that it contains a 'features' dataset with 2D shape (num_patches, feature_dim).

    Notes on using this tool:
        - The tool checks for existence of patch features H5 file as well as checks their schema for `features` dataset with shape (num_patches, feature_dim)
        - Skips files listed in `skip_wsi_to_check` and reports them.
        - Either `saveto_folder` or all of `target_magnification`, `patch_size`, and `overlap` must be specified.
            - if `saveto_folder` is None, then the tool constructs saveto_folder as `{target_magnification}x_{patch_size}px_{overlap}px_overlap`
        - Either `patch_feats_save_folder` or `patch_encoder_name` must be specified.
            - if `patch_feats_save_folder` is None, then the tool constructs patch_feats_save_folder as `features_{patch_encoder_name}`.

    Prerequisites to run this tool:
        - All files are expected in `job_dir/saveto_folder/patch_feats_save_folder/` with the name `{wsi_name}.h5`.
        - Skips filenames without extension listed in `skip_wsi_to_check` and reports them as skipped.

    The tool returns:
        dict: Mapping of WSI name (without extension) to a dict containing:
            - path: Absolute path to the feature file.
            - path_exists: True if the file exists, False otherwise.
            - file_schema_ok: True if the file contains a 2D 'features' dataset, False otherwise.
        list of str: List of absolute paths to skipped files (not WSI feature files).

    Args:
        job_dir: Path to the root directory for patch feature files.
        wsis: List of WSI names (without file extensions).
        target_magnification: Optional Patch extraction magnification. Default: 20
        patch_size: Optional Patch size in pixels. Default: 512
        overlap: Optional Patch overlap in pixels. Default: 0
        patch_encoder_name: Optional Name of the patch encoder. Default: uni_v1
        skip_wsi_to_check: Optional List of WSI names (without extension) to skip for checking. Default: None
        saveto_folder: Optional folder name to use for saving patch features. Default: None.
                    If None, tool constructs it as `{target_magnification}x_{patch_size}px_{overlap}px_overlap`.
        patch_feats_save_folder: Optional folder name for patch features. Default: None.
                    If None, tool constructs it as `features_{patch_encoder_name}`.
    """
    if skip_wsi_to_check is None:
        skip_wsi_to_check = []

    assert len(wsis) > 0, "wsis cannot be empty. Must provide WSI names without extensions to check."

    if saveto_folder is None:
        assert target_magnification is not None and patch_size is not None and overlap is not None, (
            "Either saveto_folder must be provided, or all of target_magnification, patch_size, and overlap must be specified."
        )

    if patch_feats_save_folder is None:
        assert patch_encoder_name is not None, (
            "patch_encoder_name must be specified if patch_feats_save_folder is not provided."
        )

    saveto_folder = saveto_folder or f'{target_magnification}x_{patch_size}px_{overlap}px_overlap'
    patch_feats_save_folder = patch_feats_save_folder or f'features_{patch_encoder_name}'

    # construct the directory path for patch features
    dir_with_patch_features = Path(job_dir) / saveto_folder / patch_feats_save_folder
    results = {}
    skipped = []

    for wsi_name in wsis:
        if wsi_name in skip_wsi_to_check:
            skipped.append(str(dir_with_patch_features / f"{wsi_name}.h5"))
            continue

        file_path = dir_with_patch_features / f"{wsi_name}.h5"
        result = {
            "path": str(file_path.resolve()),
            "path_exists": file_path.exists(),
            "file_schema_ok": False,
        }

        if result["path_exists"]:
            try:
                with h5py.File(file_path, 'r') as f:
                    if "features" in f:
                        ds = np.array(f["features"])
                        if hasattr(ds, "shape") and len(ds.shape) == 2:
                            result["file_schema_ok"] = True
            except Exception:
                result["file_schema_ok"] = False

        results[wsi_name] = result

    return results, skipped


@tool
def dataset_of_wsi_check_slide_features_exist_and_schema_tool(
    job_dir: str,
    wsis: list[str],
    target_magnification: int = 20,
    patch_size: int = 512,
    overlap: int = 0,
    saveto_folder: str | None = None,
    slide_encoder_name: str = "titan",
    skip_wsi_to_check: list[str] | None = None,
    slide_feats_save_folder: str | None = None,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """
    Check for the existence and schema of slide feature H5 files for a list WSIs, with support for skipping selected WSIs.

    For each WSI name in `wsis`, this tool checks whether a corresponding slide-level feature file (`{job_dir}/{saveto_folder}/{slide_feats_save_folder}/{wsi_name}.h5`) exists in the expected directory and validates that it contains a 'features' dataset with 1D shape (slide_embedding_feature_dim, 1).

    Notes on using this tool:
        - The tool checks for existence of slide features H5 file as well as checks their schema for `features` dataset with shape (slide_embedding_feature_dim, 1)
        - Skips files without extension listed in `skip_wsi_to_check` and reports them.
        - Either `saveto_folder` or all of `target_magnification`, `patch_size`, and `overlap` must be specified.
            - if `saveto_folder` is None, then the tool constructs saveto_folder as `{target_magnification}x_{patch_size}px_{overlap}px_overlap`
        - Either `slide_feats_save_folder` or `slide_encoder_name` must be specified.
            - if `slide_feats_save_folder` is None, then the tool constructs slide_feats_save_folder as `slide_features_{slide_encoder_name}`.

    Prerequisites to run this tool:
        - The tool only checks for files in `job_dir/saveto_folder/slide_feats_save_folder/` with the name `{wsi_name}.h5`.
        - The skip list should contain WSI filenames without extensions.

    The tool returns:
        dict: Mapping of WSI name (without extension) to a dict containing:
            - path: Absolute path to the feature file.
            - path_exists: True if the file exists, False otherwise.
            - file_schema_ok: True if the file contains a 1D 'features' dataset, False otherwise.
        list of str: List of absolute paths to skipped files (not WSI feature files).

    Args:
        job_dir: Path to the root directory for slide feature files.
        wsis: List of WSI names (without file extensions).
        target_magnification: Optional Patch extraction magnification. Default: 20
        patch_size: Optional Patch size in pixels. Default: 512
        overlap: Optional Patch overlap in pixels. Default: 0
        slide_encoder_name: Optional Name of the slide encoder. Default: 'titan'
        skip_wsi_to_check: Optional List of WSI names (without extension) to skip for checking. Default: None (all items in `wsis` checked).
        saveto_folder: Optional folder name to use for saving slide features. Default: None
                    If None, tool constructs it as `{target_magnification}x_{patch_size}px_{overlap}px_overlap`.
        slide_feats_save_folder: Optional folder name for slide features. Default: None.
                    If None, tool constructs it as `slide_features_{slide_encoder_name}`.
    """
    if skip_wsi_to_check is None:
        skip_wsi_to_check = []

    assert len(wsis) > 0, "wsis cannot be empty. Must provide WSI names without extensions to check."

    if saveto_folder is None:
        assert target_magnification is not None and patch_size is not None and overlap is not None, (
            "Either saveto_folder must be provided, or all of target_magnification, patch_size, and overlap must be specified."
        )

    if slide_feats_save_folder is None:
        assert slide_encoder_name is not None, (
            "slide_encoder_name must be specified if slide_feats_save_folder is not provided."
        )

    saveto_folder = saveto_folder or f'{target_magnification}x_{patch_size}px_{overlap}px_overlap'
    slide_feats_save_folder = slide_feats_save_folder or f'slide_features_{slide_encoder_name}'
    dir_with_slide_features = Path(job_dir) / saveto_folder / slide_feats_save_folder
    results = {}
    skipped = []

    for wsi_name in wsis:
        if wsi_name in skip_wsi_to_check:
            # Note: We skip checking existence/schema for these, but collect their would-be path
            skipped.append(str(dir_with_slide_features / f"{wsi_name}.h5"))
            continue

        file_path = dir_with_slide_features / f"{wsi_name}.h5"
        result = {
            "path": str(file_path.resolve()),
            "path_exists": file_path.exists(),
            "file_schema_ok": False,
        }

        if result["path_exists"]:
            try:
                with h5py.File(file_path, 'r') as f:
                    if "features" in f:
                        ds = np.array(f["features"])
                        if hasattr(ds, "shape") and len(ds.shape) == 1:
                            result["file_schema_ok"] = True
            except Exception:
                result["file_schema_ok"] = False

        results[wsi_name] = result

    return results, skipped

from pathlib import Path
from typing import Any

from wsidata import WSIData


def _write_wsi_to_zarr_store(wsi: WSIData, wsi_store_path: Path) -> None:
    """
    Write the WSI object to a zarr store at the specified path.

    Args:
        wsi (WSIData): The WSI object to write.
        wsi_store_path (str): Path to the zarr store where the WSI will be saved.

    Raises:
        RuntimeError: If writing to the zarr store fails.

    Notes:
        - Assumes the `WSIData` object implements a `.write()` method that
          accepts the target zarr store path.
        - The function will overwrite the target if it already exists.
    """
    try:
        wsi.write(file_path=wsi_store_path)
    except Exception as e:
        raise RuntimeError(
            (f"Failed to write WSI zarr store to '{wsi_store_path}'. Original error: {e}"),
        ) from e


def _paths_to_str(obj: Any) -> dict[str, Any] | list[Any] | str:
    """
    Recursively convert all pathlib.Path objects in a nested data structure to strings.

    Traverses dictionaries and lists, converting any Path objects found at any depth
    into their string representations. All other types are returned unchanged.

    Args:
        obj: A dictionary, list, Path object, or other Python object. May be arbitrarily nested.

    Returns:
        The same structure as `obj`, but with all Path objects replaced by strings.

    Notes:
        - Only dict, list, and Path objects are traversed or converted. All other types
          are left unchanged.
        - Useful for preparing data structures for JSON serialization or for passing
          data to APIs that require file paths as strings.
        - This function does not modify the original object in-place; a new structure
          is returned.
    """
    if isinstance(obj, dict):
        return {k: _paths_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_paths_to_str(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def _prepare_paths(wsi_path: str, job_dir: str) -> dict:
    """
    Prepare and create working directories and key file paths for a WSI processing job.

    Given the path to a whole slide image (WSI) file and a base job directory, this function:
      - Parses the WSI file name,
      - Creates a job-specific subdirectory,
      - Constructs the path for a zarr store output file,
      - Ensures the job directory exists on disk.

    Args:
        wsi_path (str): Path to the WSI file.
        job_dir (str): Root directory for job outputs.

    Returns:
        dict: Dictionary containing:
            - 'wsi_path_obj' (Path): Path object for the original WSI file.
            - 'wsi_name' (str): Stem (filename without extension) of the WSI file.
            - 'wsi_job_dir' (Path): Path object for the job-specific output directory.
            - 'wsi_store_path' (Path): Path object for the expected zarr store file.

    Raises:
        OSError: If the output directory cannot be created.

    Notes:
        - All returned paths are `pathlib.Path` objects.
        - The job directory is created if it does not already exist.
    """
    wsi_path_obj = Path(wsi_path)
    wsi_name = wsi_path_obj.stem
    wsi_job_dir = Path(job_dir) / wsi_name
    wsi_store_path = wsi_job_dir / f"{wsi_name}.zarr"
    wsi_job_dir.mkdir(parents=True, exist_ok=True)

    return {
        "wsi_path_obj": wsi_path_obj,
        "wsi_name": wsi_name,
        "wsi_job_dir": wsi_job_dir,
        "wsi_store_path": wsi_store_path,
    }

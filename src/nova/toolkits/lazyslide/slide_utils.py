import warnings
from typing import Any

import pandas as pd
from wsidata import WSIData

warnings.filterwarnings("ignore")


def get_slide_pyramid_info(wsi: WSIData) -> dict[str, Any]:
    """ """

    properties = wsi.properties

    # Basic info
    n_levels = properties.n_level
    base_height, base_width = properties.shape
    level_downsamples = properties.level_downsample
    base_magnification = properties.magnification
    base_mpp = properties.mpp

    levels = {}
    for level in range(n_levels):
        downsample = level_downsamples[level]
        level_height = int(base_height / downsample)
        level_width = int(base_width / downsample)
        info = {
            "level": level,
            "img_height_at_level": level_height,
            "img_width_at_level": level_width,
            "downsample_to_level_wrt_level0": downsample,
        }
        if base_magnification:
            info["magnification_at_level"] = base_magnification / downsample
        if base_mpp:
            info["mpp_at_level"] = base_mpp * downsample
        levels[f"level_{level}"] = info

    return {
        "n_levels": n_levels,
        "base_height": base_height,
        "base_width": base_width,
        "base_magnification": base_magnification,
        "base_mpp": base_mpp,
        "levels": levels,
    }


def geometry_to_patch_coords(wsi, tile_key, geometry):
    """
    Extracts base image and spatial coordinates for a patch from its geometry.

    Given a WSI object, a tile key, and a geometry (polygon), this function returns a dictionary
    containing the base image dimensions, pyramid level, and the top-left coordinates of the patch.

    Args:
        wsi (WSIData): The WSI object containing slide metadata and tile specifications.
        tile_key (str): The key used to access the tile specification from the WSI.
        geometry (shapely.geometry.Polygon): The geometry of the patch (polygon object).

    Returns:
        dict: A dictionary with the following keys:
            - 'base_height' (int): Height of the base image level.
            - 'base_width' (int): Width of the base image level.
            - 'base_level' (int): The base pyramid level index.
            - 'top_left_x' (int): The x-coordinate of the patch's top-left corner.
            - 'top_left_y' (int): The y-coordinate of the patch's top-left corner.

    Notes:
        - Assumes geometry has a `.bounds` property returning (minx, miny, maxx, maxy).
        - The coordinates are in pixel units at the base pyramid level.
    """
    d = wsi.tile_spec(key=tile_key).__dict__
    base_height = int(d["width"] * d["base_downsample"])
    base_width = int(d["height"] * d["base_downsample"])
    base_level = int(d["base_level"])
    minx, miny, maxx, maxy = geometry.bounds
    top_left_x = int(minx)
    top_left_y = int(miny)
    patch_dict = {
        "base_height": base_height,
        "base_width": base_width,
        "base_level": base_level,
        "top_left_x": top_left_x,
        "top_left_y": top_left_y,
    }

    return patch_dict


def patch_idx_to_patch_coord(
    wsi: WSIData,
    list_of_patch_ids: list,
    patch_features_key: str,
    tile_key: str,
) -> list:
    """
    Given a list of patch (tile) IDs, returns a list of dictionaries containing patch spatial information.

    For each patch ID, extracts its top-left coordinates and metadata such as base image dimensions and level
    from the WSI and tile data using the `geometry_to_patch_coords` helper. Skips any IDs not present in the WSI.

    Args:
        wsi (WSIData): The WSIData object containing the patch and tile information.
        list_of_patch_ids (list): List of patch (tile) IDs for which to extract coordinates.
        patch_features_key (str): Key to retrieve patch features AnnData from the WSI object.
        tile_key (str): Key to retrieve tile geometry DataFrame from the WSI object.

    Returns:
        list: A list of dictionaries, one per found patch, with the following keys:
            - 'patch_id': The patch (tile) ID.
            - 'base_height': The base height of the WSI (pixels at base level).
            - 'base_width': The base width of the WSI (pixels at base level).
            - 'base_level': The base pyramid level of the WSI.
            - 'top_left_x': The x-coordinate (pixel) of the patch's top-left corner.
            - 'top_left_y': The y-coordinate (pixel) of the patch's top-left corner.

    Raises:
        AssertionError: If no patches in `list_of_patch_ids` are found in the WSI.

    Notes:
        - The returned coordinates are in pixel units at the base pyramid level.
        - Skips any patch IDs not present in the merged tile-feature data.
    """
    adata = wsi[patch_features_key]
    df = adata.obs
    assert isinstance(df, pd.DataFrame), f"Expected DataFrame for patch features, got {type(df)=}"
    gdf = wsi[tile_key]
    merged = gdf.merge(df, on="tile_id")

    patch_coords_list = []
    for patch_id in list_of_patch_ids:
        patch_id = int(patch_id)
        row = merged[merged["tile_id"] == patch_id]
        if row.empty:
            continue
        geometry = row.iloc[0]["geometry"]

        patch_dict = geometry_to_patch_coords(wsi, tile_key, geometry)
        patch_dict["patch_id"] = patch_id
        patch_coords_list.append(patch_dict)

    assert len(patch_coords_list) > 0, (
        f"No patches found for patch_ids: {list_of_patch_ids} in WSI {getattr(wsi, 'name', 'unknown')}"
    )
    return patch_coords_list

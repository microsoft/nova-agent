from typing import Tuple

import scanpy as sc
from wsidata import WSIData


def check_tissue_key_in_WSI(wsi: WSIData, tissue_key: str) -> Tuple[bool, str]:
    """
    Check if the specified tissue key exists in the WSI data.

    Verifies that the `shapes` attribute is present in the WSI object and
    that `tissue_key` is one of the available keys. Raises a KeyError
    if the key is missing or if `shapes` is not set.

    Args:
        wsi (WSIData): The whole slide image object to check.
        tissue_key (str): The tissue key to look for within the WSI's shapes.

    Raises:
        KeyError: If `shapes` is not present or `tissue_key` is not found.
    """
    shapes = getattr(wsi, "shapes", None)
    if shapes is None or tissue_key not in shapes:
        error_msg = f"""
        Provided tissue_key '{tissue_key}' not found in WSI data.
        Available keys: {list(shapes.keys()) if shapes else 'None'}.
        Please run tissue segmentation first using the tissue_key {tissue_key}.
        """
        return False, error_msg

    return True, f'successfully found tissue key {tissue_key} in WSI data'


def check_valid_tissue_id_in_WSI(wsi: WSIData, tissue_key: str, tissue_id: int) -> Tuple[bool, str]:
    """
    Check if the specified tissue ID exists in the WSI data for a given tissue key.
    """
    valid_tissue_ids = wsi[tissue_key]['tissue_id'].unique().tolist()

    if valid_tissue_ids is None or len(valid_tissue_ids) == 0:
        error_msg = f"""
        No valid tissue IDs found in WSI for key '{tissue_key}'.
        Please ensure tissue segmentation has been performed with the key '{tissue_key}'.
        """
        return False, error_msg

    if tissue_id not in valid_tissue_ids:
        error_msg = f"""
        Specified tissue_id '{tissue_id}' is not valid for WSI.
        Valid tissue IDs are: {valid_tissue_ids}.
        """
        return False, error_msg

    return True, ''


def check_tile_key_in_WSI(wsi_data: WSIData, tile_key: str) -> Tuple[bool, str]:
    """
    Check if the specified tile key exists in the WSI data.

    Verifies that the `shapes` attribute is present in the WSI object and
    that `tile_key` is one of the available keys.

    Args:
        wsi_data: The whole slide image object to check.
        tile_key (str): The tile key to look for within the WSI's shapes.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if the key exists, False otherwise
            - str: Empty string if successful, error message if failed
    """
    shapes = getattr(wsi_data, "shapes", None)
    if shapes is None or tile_key not in shapes:
        error_msg = f"""
        Provided tile_key '{tile_key}' not found in WSI data.
        Available keys: {list(shapes.keys()) if shapes else 'None'}.
        Please run tile extraction first using the tile_key {tile_key}.
        """
        return False, error_msg

    return True, ''


def check_patch_features_key_in_WSI(wsi_data: WSIData, patch_features_key: str) -> Tuple[bool, str]:
    """
    Check if the specified patch features key exists in the WSI data tables.

    Verifies that the `patch_features_key` is present in the `tables` attribute
    of the WSI object. Raises a KeyError if the key is not found.

    Args:
        wsi_data: The whole slide image object to check.
        patch_features_key (str): The key to look for within the WSI's tables.

    Raises:
        KeyError: If `patch_features_key` is not found in `wsi_data.tables`.
    """
    if patch_features_key not in wsi_data.tables:
        error_msg = f"""
        Provided patch_features_key '{patch_features_key}' not found in WSI data tables.
        Available keys: {list(wsi_data.tables.keys()) if wsi_data.tables else 'None'}.
        Please run patch feature extraction first using the key {patch_features_key}.
        """
        return False, error_msg
    return True, ''


def check_slide_features_key_in_WSI(
    wsi_data: WSIData, patch_features_key: str, slide_features_key: str
) -> Tuple[bool, str]:
    """


    Args:
        wsi_data: The whole slide image object to check.
        slide_features_key (str): The key to look for within the WSI's tables.

    Raises:
        KeyError: If `slide_features_key` is not found in `wsi_data[patch_features_key].varm`.
    """
    if slide_features_key not in wsi_data[patch_features_key].varm.keys():
        error_msg = f"""
        Provided slide_features_key '{slide_features_key}' not found in WSI data.
        Available keys: {list(wsi_data[patch_features_key].varm.keys()) if wsi_data[patch_features_key].varm else 'None'}.
        Please run slide feature extraction first using the key {slide_features_key}.
        """
        return False, error_msg
    return True, ''


def check_reduction_key_in_WSI(adata: sc.AnnData, key: str = "X_pca") -> Tuple[bool, str]:
    """
    Check if a specific reduction key exists in the AnnData object's `.obsm` attribute.

    Args:
        adata (AnnData): The AnnData object to check.
        key (str, optional): The key to check in `adata.obsm`. Defaults to "X_pca".

    Raises:
        RuntimeError: If the key is not present in `adata.obsm`, or if `.obsm` is missing or empty.
    """
    if not hasattr(adata, "obsm") or not adata.obsm or key not in adata.obsm:
        error_msg = f"""
        Provided key '{key}' not found in AnnData.obsm.
        Available keys: {list(adata.obsm.keys()) if adata.obsm else 'None'}.
        Please run feature space reduction first using the key {key}.
        """
        return False, error_msg
    return True, ''


def check_clustering_key_in_WSI(adata: sc.AnnData, key: str = "leiden_0.75") -> Tuple[bool, str]:
    """
    Check if a specific clustering key exists in the AnnData object's `.obs` attribute.

    Args:
        adata (AnnData): The AnnData object to check.
        key (str, optional): The key to check in `adata.obs`. Defaults to "leiden_0.75".

    Raises:
        RuntimeError: If the key is not present in `adata.obs`, or if `.obs` is missing or empty.
    """
    if not hasattr(adata, "obs") or key not in adata.obs.columns:
        error_msg = (
            f"Provided key '{key}' not found in AnnData.obs.\n"
            f"Available keys: {list(adata.obs.columns)}.\n"
            f"Please run clustering first using the key '{key}'."
        )
        return False, error_msg
    return True, ''

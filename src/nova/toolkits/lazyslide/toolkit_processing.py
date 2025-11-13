import json
from typing import Any

from wsidata import open_wsi

import lazyslide as zs
from nova.toolkits.lazyslide.slide_utils import (
    get_slide_pyramid_info,
)
from nova.toolkits.lazyslide.toolkit_base import TILES_KEY, TISSUES_KEY, LazySlideBaseToolKit
from nova.utils.io import _write_wsi_to_zarr_store
from nova.utils.summarize import print_log


class LazySlideProcessingToolKit(LazySlideBaseToolKit):
    def retrieve_properties_from_wsi(
        self,
        tile_key: str | None = None,
        tissue_key: str | None = None,
        patch_features_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve and serialize slide properties and pyramid information from a WSI file.

        Loads a WSI object, gathers its base and pyramid properties, optionally includes
        tissue and tile counts, and writes the information to a JSON file.

        Args:
            tile_key (Optional[str], optional): Tile key for fetching tile counts, if present.
            tissue_key (Optional[str], optional): Tissue key for fetching tissue counts, if present.

        Returns:
            dict: Dictionary containing slide and pyramid properties, including:
            - 'n_levels' (int): Number of pyramid levels.
            - 'base_height' (int): Height at level 0.
            - 'base_width' (int): Width at level 0.
            - 'base_magnification' (float or None): Magnification at level 0.
            - 'base_mpp' (float or None): Microns per pixel at level 0.
            - 'levels' (dict): Information for each pyramid level.
            - 'wsi_file_path' (str): Path to the WSI file.
            - 'wsi_reader_name' (str): Name of the WSI reader backend.
            - 'raw_properties' (dict): Raw properties of the WSI.
            - 'n_tissues' (int, optional): Number of tissue regions (if tissue_key is provided).
            - 'n_tiles' (int, optional): Number of tiles (if tile_key is provided).
            - 'asset_dict_path' (str): Path to the output JSON file.

        Raises:
            RuntimeError: If loading the WSI, checking keys, or writing the output JSON fails.
        """

        log = []

        # we load from src if it is there. if it is not we load a new WSI.
        # check if zarr exists?
        if self.wsi_store_path.exists():
            wsi = open_wsi(self.wsi_path, store=str(self.wsi_store_path))
            log.append(f"\n Loaded WSI successfully from source {self.wsi_store_path}.")
        else:
            wsi = open_wsi(self.wsi_path)
            log.append("\n Loaded WSI successfully.")

        # Collect the basic properties and pyramid information
        asset_dict = get_slide_pyramid_info(wsi)

        # File and reader
        asset_dict["wsi_file_path"] = str(self.wsi_path)
        asset_dict["wsi_reader_name"] = wsi.reader.name
        asset_dict["raw_properties"] = wsi.properties.raw

        # Additional properties if available
        if tissue_key:
            log = self._check_tissue_key(wsi, tissue_key, log)
            n_tissues = wsi.fetch.n_tissue(tissue_key)
            asset_dict["n_tissues"] = n_tissues

        if tile_key:
            log = self._check_tile_key(wsi, tile_key, log)
            n_tiles = wsi.fetch.n_tiles(tile_key)
            asset_dict[f"n_tiles_for_{tile_key}"] = n_tiles

        if patch_features_key:
            log = self._check_patch_features_key(wsi, patch_features_key, log)
            asset_dict[f"{patch_features_key}_exists"] = True
            asset_dict[f"{patch_features_key}_embedding_dim"] = wsi[patch_features_key].X.shape[1]  # type: ignore

        # Write asset_dict to JSON file
        asset_dict_path = self.wsi_job_dir / f"{self.wsi_name}_properties.json"
        asset_dict["asset_dict_path"] = str(asset_dict_path)
        with open(asset_dict_path, "w") as f:
            json.dump(asset_dict, f, indent=4)

        # do closing operations
        try:
            _write_wsi_to_zarr_store(wsi, self.wsi_store_path)
            log.append(f"\n WSI written to zarr store at {self.wsi_store_path}.")

            asset_dict["operations_log"] = print_log(log)

        except Exception as e:
            raise RuntimeError(
                f"Failed to write WSI to zarr store. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        return asset_dict

    def extract_tissue_in_wsi(
        self,
        tissue_key: str = TISSUES_KEY,
        overwrite: bool = False,
    ) -> dict[str, list[int] | str]:
        """
        Run tissue segmentation on a WSI and save the results to a Zarr store.

        Args:
            tissue_key (str): Key under which tissue segmentation results are stored.
            overwrite (bool): If True, re-runs tissue segmentation even if results exist.

        Returns:
            dict: Summary of tissue segmentation:
            - 'wsi_store_path' (str): Path to the output Zarr store file.
            - 'tissue_ids' (List[int]): List of unique tissue IDs found.
            - 'tissue_key' (str): Key under which segmentation results are stored.

        Raises:
            RuntimeError: If segmentation or writing to Zarr store fails.
        """

        log = []

        wsi, log = self._load_wsi_with_checks(log)

        # Run tissue segmentation
        if overwrite or not self.load_from_src:
            print(
                f"Overwrite requested or load_from_src is False. Re-running tissue segmentation. Results will be saved in tissue_key: {tissue_key}."
            )

            try:
                zs.seg.tissue(wsi, key_added=tissue_key)  # type: ignore
                zs.tl.tissue_props(wsi, key=tissue_key)  # type: ignore
                log.append(f"\n Tissue segmentation completed and saved at key {tissue_key}.")
            except Exception as e:
                raise RuntimeError(f"Tissue segmentation failed for '{self.wsi_path}'. Original error: {e}") from e
        else:
            print(f"Overwrite not requested. Checking if tissues exist at key {tissue_key}.\n\n", end='')
            log = self._check_tissue_key(wsi, tissue_key, log)
            print(f"Tissue segmentations exist at key {tissue_key}. They won't be updated.")

        try:
            _write_wsi_to_zarr_store(wsi, self.wsi_store_path)
            log.append(f"\n WSI written to zarr store at {self.wsi_store_path}.")

            asset_dict = {
                "wsi_store_path": str(self.wsi_store_path),
                "tissue_ids": wsi[tissue_key]['tissue_id'].unique().tolist(),
                "tissue_key": tissue_key,
                "operations_log": print_log(log),
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to write WSI to zarr store. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        return asset_dict

    def extract_coordinates_in_wsi(
        self,
        tissue_key: str = TISSUES_KEY,
        tile_key: str = TILES_KEY,
        tile_px: int = 256,
        mpp: float = 0.5,
        overlap: float = 0.0,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """
        Extract tile (patch) coordinates from a WSI and write updated data to a Zarr store.

        Args:
            tissue_key (str): Key under which tissue segmentation results are stored.
            tile_key (str): Key under which tile extraction results are stored.
            tile_px (int): Tile (patch) size in pixels.
            mpp (float): Target microns per pixel for tiling.
            overlap (float): Overlap fraction between adjacent tiles.
            overwrite (bool): If True, re-runs tissue tiling even if results exist.

        Returns:
            dict: Summary of the patch extraction:
            - 'wsi_store_path' (str): Path to the output Zarr store file.
            - 'tissue_key' (str): Key under which tissue segmentation is stored.
            - 'tile_key' (str): Key under which tile extraction is stored.
            - 'num_patches' (int): Number of patches (tiles) generated.

        Raises:
            RuntimeError: If loading the WSI, checking keys, running tiling, or writing
            to the Zarr store fails.

        Notes:
            - Assumes `zs.pp.tile_tissues` is available for patch extraction.
            - Prerequisite: tissue segmentation must exist for the provided tissue_key.
        """

        log = []

        wsi, log = self._load_wsi_with_checks(log)

        # Prerequisite: Check tissue segmentation exists
        log = self._check_tissue_key(wsi, tissue_key, log)

        # Run tile extraction if needed
        if overwrite or not self.load_from_src:
            print(f"Overwrite requested. Re-running tissue tiling. Results will be saved in tile_key {tile_key}.")
            try:
                zs.pp.tile_tissues(wsi, tile_px=tile_px, mpp=mpp, overlap=overlap, key_added=tile_key)
                log.append(f"\n Tissue tiling completed and saved at key {tile_key}.")
            except Exception as e:
                raise RuntimeError(f"""Tissue tiling failed for '{self.wsi_path}'. Original error: {e}""") from e
        else:
            print(f"Overwrite not requested. Checking if tiles already exist at key {tile_key}.\n\n", end='')
            log = self._check_tile_key(wsi, tile_key, log)
            print(f"Tissue tiles exist at key {tile_key}. They won't be updated.")

        try:
            _write_wsi_to_zarr_store(wsi, self.wsi_store_path)
            log.append(f"\n WSI written to zarr store at {self.wsi_store_path}.")

            asset_dict = {
                "wsi_store_path": str(self.wsi_store_path),
                "tissue_key": tissue_key,
                "tile_key": tile_key,
                "num_patches": wsi.fetch.n_tiles(tile_key),
                "operations_log": print_log(log),
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to write WSI to zarr store. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        return asset_dict

    def extract_patch_features_in_wsi(
        self,
        patch_features_key: str | None = None,
        tissue_key: str = TILES_KEY,
        tile_key: str = TISSUES_KEY,
        model: str = "uni",
        device: str = "cuda:0",
        batch_size: int = 64,
        num_workers: int = 20,
        overwrite: bool = False,
    ) -> dict:
        """
        Extract patch-level features for a WSI and save results to a Zarr store.

        Args:
            tissue_key (str): Key under which tissue segmentation results are stored.
            tile_key (str): Key under which tile extraction results are stored.
            model (str): Name of the feature extraction model to use.
            device (str): Device on which to run inference (e.g., "cuda", "cpu").
            batch_size (int): Batch size for inference.
            num_workers (int): Number of workers for data loading.
            overwrite (bool): If True, re-runs feature extraction even if results exist.

        Returns:
            dict: Summary of feature extraction:
            - 'tissue_key' (str): Key for tissue segmentation used.
            - 'tile_key' (str): Key for tile extraction used.
            - 'patch_features_key' (str): Key under which extracted features are stored.
            - 'embedding_dim' (int): Dimensionality of the extracted patch features.

        Raises:
            RuntimeError: If loading the WSI, running feature extraction, or writing to Zarr store fails.

        Notes:
            - Prerequisite: tissue segmentation and tile extraction must already exist.
            - Feature extraction is performed using `zs.tl.feature_extraction`.
        """

        assert model in zs.models.list_models(), (  # type: ignore
            f"Model '{model}' is not supported. Available models: {zs.models.list_models()}"  # type: ignore
        )

        log = []

        wsi, log = self._load_wsi_with_checks(log)

        # Prerequisite: Check tissue segmentation exists
        log = self._check_tissue_key(wsi, tissue_key, log)

        # Prerequisite: Check tissue tiles exist
        log = self._check_tile_key(wsi, tile_key, log)

        patch_features_key = patch_features_key or f"{model}_{tile_key}"
        if overwrite or not self.load_from_src:
            print(
                f"Overwrite requested. Re-running patch feature extraction. "
                f"Results will be saved in patch_features_key {patch_features_key}.\n\n"
            )
            try:
                zs.tl.feature_extraction(  # type: ignore
                    wsi=wsi,
                    model=model,
                    device=device,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    tile_key=tile_key,
                    key_added=patch_features_key,
                    return_features=False,
                )
                log.append(f"\n Patch feature extraction completed and saved at key {patch_features_key}.")
            except Exception as e:
                raise RuntimeError(
                    f"""Feature extraction failed for '{self.wsi_path}'. Original error: {e}"""
                    f"\n Operations log: {print_log(log)}"
                ) from e
        else:
            print(
                f"Overwrite not requested. Checking if patch features already exist at key {patch_features_key}.\n\n",
                end='',
            )
            log = self._check_patch_features_key(wsi, patch_features_key, log)
            print(f"Patch features exist at key {patch_features_key}. They won't be updated.")

        try:
            _write_wsi_to_zarr_store(wsi, self.wsi_store_path)
            log.append(f"\n WSI written to zarr store at {self.wsi_store_path}.")

            asset_dict = {
                "tissue_key": tissue_key,
                "tile_key": tile_key,
                "patch_features_key": patch_features_key,
                "embedding_dim": wsi[patch_features_key].X.shape[1],  # type: ignore
                "operations_log": print_log(log),
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to write WSI to zarr store. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        return asset_dict

    def encode_single_wsi(
        self,
        slide_encoder: str = "mean",
        tissue_key: str = TISSUES_KEY,
        tile_key: str = TILES_KEY,
        patch_features_key: str = "conch_v1.5",
        slide_features_key: str | None = None,
        overwrite: bool = False,
        device: str = 'cuda:0',
    ) -> dict:
        assert slide_encoder in ['mean', 'median', 'prism', 'titan'], (  # type: ignore
            f"Model '{slide_encoder}' is not supported. Available models: {['mean', 'median', 'prism', 'titan']}"  # type: ignore
        )

        log = []

        wsi, log = self._load_wsi_with_checks(log)

        # Prerequisite: Check tissue segmentation exists
        log = self._check_tissue_key(wsi, tissue_key, log)

        # Prerequisite: Check tissue tiles exist
        log = self._check_tile_key(wsi, tile_key, log)

        # Prerequisite: Check patch features exist
        log = self._check_patch_features_key(wsi, patch_features_key, log)

        # make sure the correct patch encoder used for the specified slide encoder
        if slide_encoder == 'titan':
            assert 'conch_v1.5' in patch_features_key, (
                f"titan slide encoder requires conch_v1.5 patch features. Log: {print_log(log)}"
            )
        elif slide_encoder == 'prism':
            assert 'virchow' in patch_features_key and 'virchow2' not in patch_features_key, (
                f"prism slide encoder requires virchow patch features. Log: {print_log(log)}"
            )
        log.append(f"\n Slide encoder '{slide_encoder}' is compatible with patch features '{patch_features_key}'.")

        slide_features_key = slide_features_key or f'{slide_encoder}_slide_representation'
        if overwrite or not self.load_from_src:
            print(
                f"Overwrite requested. Re-running slide encoding. "
                f"Results will be saved in slide_features_key {slide_features_key}.\n\n"
            )
            try:
                zs.tl.feature_aggregation(  # type: ignore
                    wsi,
                    tile_key=tile_key,
                    feature_key=patch_features_key,
                    encoder=slide_encoder,
                    agg_key=slide_features_key,
                    device=device,  # type: ignore lazyslide issue
                )
                log.append(f"\n Slide encoding completed and saved at key {slide_features_key}.")
            except Exception as e:
                raise RuntimeError(
                    f"""Slide encoding failed for '{self.wsi_path}'. Original error: {e}"""
                    f"\n Operations log: {print_log(log)}"
                ) from e

        else:
            print(
                f"Overwrite not requested. Checking if slide features already exist at key {slide_features_key}.\n\n",
                end='',
            )
            log = self._check_slide_features_key(wsi, patch_features_key, slide_features_key, log)
            print(f"Slide features exist at key {slide_features_key}. They won't be updated.")

        try:
            _write_wsi_to_zarr_store(wsi, self.wsi_store_path)
            log.append(f"\n WSI written to zarr store at {self.wsi_store_path}.")

            asset_dict = {
                "tissue_key": tissue_key,
                "tile_key": tile_key,
                "patch_features_key": patch_features_key,
                "slide_features_key": slide_features_key,
                "embedding_dim": wsi[patch_features_key].varm[slide_features_key].shape[0],  # type: ignore
                "operations_log": print_log(log),
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to write WSI to zarr store. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        return asset_dict

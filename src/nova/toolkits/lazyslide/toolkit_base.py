from pathlib import Path

from anndata import AnnData
from wsidata import WSIData, open_wsi

from nova.toolkits.lazyslide.checks import (
    check_clustering_key_in_WSI,
    check_patch_features_key_in_WSI,
    check_reduction_key_in_WSI,
    check_slide_features_key_in_WSI,
    check_tile_key_in_WSI,
    check_tissue_key_in_WSI,
    check_valid_tissue_id_in_WSI,
)
from nova.utils.summarize import print_log

TISSUES_KEY = "tissues"
TILES_KEY = "tissue_tiles"
PATCH_FEATURES_KEY = "uni_tissue_tiles"


# TODO: can provide wsi_store_path as an argument to the class
class LazySlideBaseToolKit:
    def __init__(
        self,
        wsi_path: str,
        job_dir: str,
        load_from_src: bool = False,
    ) -> None:
        """
        Args:
            wsi_path (str): The file path to the whole slide image (WSI).
            job_dir (str): The directory where job-related files will be stored.
            load_from_src (bool, optional): Flag indicating whether to load data from the source.
                Defaults to False.
        """
        self.wsi_path = Path(wsi_path)
        self.wsi_job_dir = Path(job_dir) / self.wsi_name
        self.wsi_job_dir.mkdir(parents=True, exist_ok=True)
        self.load_from_src = load_from_src

    @property
    def wsi_name(self) -> str:
        return self.wsi_path.stem

    @property
    def wsi_store_path(self) -> Path:
        wsi_store_path = self.wsi_job_dir / f"{self.wsi_name}.zarr"
        return wsi_store_path

    def _load_wsi_with_checks(
        self,
        log: list[str],
    ) -> tuple[WSIData, list[str]]:
        """
        Load a Whole Slide Image (WSI) with optional checks for a pre-existing Zarr store.

        Returns:
            WSIData: The loaded Whole Slide Image.

        Raises:
            FileNotFoundError: If `load_from_src` is True but `wsi_store_path` does not exist.
            RuntimeError: If loading the WSI fails.
        """

        if self.load_from_src and self.wsi_store_path:
            if not self.wsi_store_path.exists():
                raise FileNotFoundError(
                    f"""Requested to load from existing zarr file, but none found at {self.wsi_store_path}.
                    Run with load_from_src=False or ensure the zarr file is present."""
                    f" Operation log: {print_log(log)}"
                )
            try:
                wsi = open_wsi(self.wsi_path, store=str(self.wsi_store_path))
                log.append(f"\n Loaded WSI successfully from source {self.wsi_store_path}.")
            except Exception as e:
                raise RuntimeError(
                    f"""Failed to open WSI at '{self.wsi_path}' (store: '{self.wsi_store_path}'). Original error: {e}"""
                    f" Operation log: {print_log(log)}"
                ) from e
        else:
            try:
                wsi = open_wsi(self.wsi_path)
                log.append("\n Loaded WSI successfully.")
            except Exception as e:
                raise RuntimeError(
                    f"""Failed to open WSI at '{self.wsi_path}'. Original error: {e}"""
                    f" Operation log: {print_log(log)}"
                ) from e
        return wsi, log

    def _check_tissue_key(
        self,
        wsi: WSIData,
        tissue_key: str,
        log: list[str],
    ) -> list[str]:
        tissue_seg_status, tissue_seg_error = check_tissue_key_in_WSI(wsi, tissue_key)
        if tissue_seg_status:
            log.append(f"\n Found tissue segmentation in WSI at tissue_key {tissue_key}.")
        else:
            raise KeyError(tissue_seg_error + f" Operations log: {print_log(log)}")

        return log

    def _check_tile_key(
        self,
        wsi: WSIData,
        tile_key: str,
        log: list[str],
    ) -> list[str]:
        tile_status, tile_error = check_tile_key_in_WSI(wsi, tile_key)
        if tile_status:
            log.append(f"\n Found tiles in WSI at tile_key {tile_key}.")
        else:
            raise KeyError(tile_error + f" Operations log: {print_log(log)}")

        return log

    def _check_valid_tissue_id(self, wsi: WSIData, tissue_id: int | None, tissue_key: str, log: list[str]) -> list[str]:
        if tissue_id is not None:
            tissue_id_status, tissue_id_error = check_valid_tissue_id_in_WSI(wsi, tissue_key, tissue_id)
            if tissue_id_status:
                log.append(f"\n Found valid tissue_id {tissue_id} in the WSI.")
            else:
                raise KeyError(tissue_id_error + f" Operations log: {print_log(log)}")

        return log

    def _check_patch_features_key(
        self,
        wsi: WSIData,
        patch_features_key: str,
        log: list[str],
    ) -> list[str]:
        patch_features_status, patch_features_error = check_patch_features_key_in_WSI(wsi, patch_features_key)
        if patch_features_status:
            log.append(f"\n Found patch features in WSI at key {patch_features_key}.")
        else:
            raise KeyError(patch_features_error + f" Operations log: {print_log(log)}")

        return log

    def _check_slide_features_key(
        self,
        wsi: WSIData,
        patch_features_key: str,
        slide_features_key: str,
        log: list[str],
    ) -> list[str]:
        slide_features_status, slide_features_error = check_slide_features_key_in_WSI(
            wsi, patch_features_key, slide_features_key
        )
        if slide_features_status:
            log.append(f"\n Found slide features in WSI at key {slide_features_key}.")
        else:
            raise KeyError(slide_features_error + f" Operations log: {print_log(log)}")

        return log

    def _check_clustering_key(
        self,
        adata: AnnData,
        clustering_key: str,
        log: list[str],
    ) -> list[str]:
        clustering_status, clustering_error = check_clustering_key_in_WSI(adata, clustering_key)
        if clustering_status:
            log.append(f"\n Found clustering in WSI at key {clustering_key}.")
        else:
            raise KeyError(clustering_error + f" Operations log: {print_log(log)}")

        return log

    def _check_reduction_key(
        self,
        adata: AnnData,
        reduction_key: str,
        log: list[str],
    ) -> list[str]:
        reduction_status, reduction_error = check_reduction_key_in_WSI(adata, reduction_key)
        if reduction_status:
            log.append(f"\n Found reduction in WSI at key {reduction_key}.")
        else:
            raise KeyError(reduction_error + f" Operations log: {print_log(log)}")

        return log

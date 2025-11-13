import json
from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image

import lazyslide as zs
from nova.toolkits.lazyslide.slide_utils import (
    patch_idx_to_patch_coord,
)
from nova.toolkits.lazyslide.toolkit_base import (
    PATCH_FEATURES_KEY,
    TILES_KEY,
    TISSUES_KEY,
    LazySlideBaseToolKit,
)
from nova.utils.clustering import reduce_data, run_leiden_clustering
from nova.utils.io import _write_wsi_to_zarr_store
from nova.utils.summarize import print_log


class LazySlideVisualizationToolKit(LazySlideBaseToolKit):
    def visualize_wsi(
        self,
        tissue_key: str = TISSUES_KEY,
        tile_key: str = TILES_KEY,
        tissue_id: int | None = None,
        add_contours: bool = False,
        add_tiles: bool = False,
        color_tiles_by: str | None = None,
        patch_features_key: str | None = None,
        dpi: int = 200,
        title: str | None = None,
        fig_size: tuple[int, int] = (8, 8),
    ) -> dict[str, str]:
        """
        Visualizes a WSI. Prerequisite is tissue segmentation, which must be run before visualization.
        Optionally add the contours of the tissue segmentation and tile boundaries.
        Base functionality is showing the tissue regions.

        Args:
            tissue_key: Key for tissue segmentation results. Defaults to "tissues".
            tile_key: Key for tile extraction results. Defaults to "tissue_tiles".
            tissue_id: If specified, only show the given tissue ID. Defaults to None.
            add_contours: If True, add tissue segmentation contours. Defaults to False.
            add_tiles: If True, overlay tile boundaries. Defaults to False.
            fig_size: Figure size in inches (width, height). Defaults to (8, 8).
            dpi: Dots per inch for the output image. Defaults to 200.
            title: Custom title for the visualization. If None, an automatic
                title is generated. Defaults to None.

        Returns:
            dict: Dictionary containing:
            - 'visualization_save_path': Path to the saved PNG image.
            - 'operations_log': Log of operations performed during visualization.
        """

        assert color_tiles_by is None or patch_features_key is not None, (
            "If 'color_tiles_by' is provided, 'patch_features_key' must also be specified."
        )
        log = []

        # Set the visualization save path
        visualization_save_path = self.wsi_job_dir / f"{self.wsi_name}_overview.png"

        wsi, log = self._load_wsi_with_checks(log)

        # Check tissue segmentation exists
        if add_contours:
            log = self._check_tissue_key(wsi, tissue_key, log)

        # Verify tissue ID if specified
        if add_tiles:
            log = self._check_valid_tissue_id(wsi, tissue_id, tissue_key, log)

        if color_tiles_by is not None and patch_features_key is not None:
            log = self._check_patch_features_key(wsi, patch_features_key, log)
            adata = wsi[patch_features_key]
            log = self._check_clustering_key(adata, color_tiles_by, log)  # type: ignore

        # Create visualization figure
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

        # Add base tissue visualization
        title = f"{self.wsi_name} Tissue IDs shown: {'All' if tissue_id is None else tissue_id}"
        zs.pl.tissue(  # type: ignore
            wsi,
            ax=ax,
            tissue_id=tissue_id,
            tissue_key=tissue_key,
            show_contours=False,
            show_id=False,
            mark_origin=True,
            scalebar=True,
            title=title,
        )
        log.append("\n Added the tissues to WSI successfully.")

        # Add contours if requested
        if add_contours:
            try:
                zs.pl.tissue(  # type: ignore
                    wsi,
                    ax=ax,
                    tissue_id=tissue_id,
                    tissue_key=tissue_key,
                    show_contours=True,
                    show_id=True,
                    mark_origin=True,
                    scalebar=True,
                    title=title,
                )
                log.append(f"\n Added the contours to WSI successfully with tissue_key {tissue_key}.")
            except Exception:
                plt.close(fig)
                raise RuntimeError(
                    f"Failed to add contours visualization for WSI '{self.wsi_name}'. "
                    f"Ensure the WSI object contains the necessary data. Operations log: {print_log(log)}"
                ) from None

        # Add tiles if requested
        if add_tiles:
            log = self._check_tile_key(wsi, tile_key, log)
            try:
                zs.pl.tiles(  # type: ignore
                    wsi,
                    ax=ax,
                    tissue_id=tissue_id,
                    tissue_key=tissue_key,
                    tile_key=tile_key,
                    show_contours=True,
                    show_id=True,
                    linewidth=0.5,
                    mark_origin=True,
                    scalebar=True,
                    title=title,
                )
                log.append(f"\n Added the tiles visualization overlay to WSI successfully for tile_key {tile_key}.")
            except Exception:
                raise RuntimeError(
                    f"Failed to add tiles visualization for WSI '{self.wsi_name}'. "
                    f"Ensure the WSI object contains the necessary data. Operations log: {print_log(log)}"
                ) from None

        # if wanting to color tiles by a score column
        # different config of tiles for categorical and continuous scores
        # TODO
        if color_tiles_by is not None and patch_features_key is not None:
            is_categorical = adata.obs[color_tiles_by].dtype == 'object'

            if is_categorical:
                try:
                    # legend will have categories provided in adata.obs[color_tiles_by]
                    num_cats = len(adata.obs[color_tiles_by].unique())
                    palette = sns.color_palette("tab20", num_cats).as_hex()
                    palette_map = {cat: color for cat, color in zip(adata.obs[color_tiles_by].unique(), palette)}

                    zs.pl.tiles(  # type: ignore
                        wsi,
                        feature_key=patch_features_key,
                        tile_key=tile_key,
                        color=color_tiles_by,
                        alpha=0.5,
                        palette=palette_map,
                        show_contours=False,
                        show_id=False,
                        mark_origin=True,
                        scalebar=True,
                        ax=ax,
                    )
                    log.append(
                        f"\n Colored tiles by categorical feature '{color_tiles_by}' with {num_cats} categories."
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to color tiles by categorical feature '{color_tiles_by}' for WSI '{self.wsi_name}'. "
                        f"Original error: {e}"
                        f"Operations log: {print_log(log)}"
                    ) from e
            else:
                try:
                    # legend will have continuous values and title will be set to color_tiles_by
                    zs.pl.tiles(  # type: ignore
                        wsi,
                        feature_key=patch_features_key,
                        tile_key=tile_key,
                        color=color_tiles_by,
                        alpha=0.5,
                        show_contours=False,
                        show_id=False,
                        mark_origin=True,
                        scalebar=True,
                        cmap='inferno',
                        legend_kws={
                            'title': color_tiles_by,
                        },
                        ax=ax,
                    )
                    log.append(f"\n Colored tiles by continuous feature '{color_tiles_by}'.")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to color tiles by continuous feature '{color_tiles_by}' for WSI '{self.wsi_name}'. "
                        f"Original error: {e}"
                        f"Operations log: {print_log(log)}"
                    ) from e

        fig.savefig(visualization_save_path, bbox_inches="tight")
        plt.close(fig)
        log.append(f"\n Saved visualization image to {visualization_save_path}.")

        # do closing operations
        try:
            _write_wsi_to_zarr_store(wsi, self.wsi_store_path)
            wsi.close()
            log.append(f"\n WSI written to zarr store at {self.wsi_store_path}.")

            asset_dict = {"visualization_save_path": str(visualization_save_path), "operations_log": print_log(log)}

        except Exception as e:
            raise RuntimeError(
                f"Failed to write WSI to zarr store. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        return asset_dict

    def feature_space_reduction(
        self,
        tissue_key: str = TISSUES_KEY,
        tile_key: str = TILES_KEY,
        patch_features_key: str = PATCH_FEATURES_KEY,
        scale: bool = True,
        compute_neighbors: bool = True,
        neighbors_key_added: str = 'neighbors',
        compute_pca: bool = False,
        pca_key_added: str = 'X_pca',
        compute_umap: bool = False,
        umap_key_added: str = 'X_umap',
        compute_tsne: bool = False,
        tsne_key_added: str = 'X_tsne',
    ) -> dict:
        """
        Reduce the feature space of the WSI by applying various dimensionality reduction techniques.

        :param tissue_key: Key for the tissue data, defaults to TISSUES_KEY
        :param tile_key: Key for the tile data, defaults to TILES_KEY
        :param patch_features_key: Key for the patch features, defaults to PATCH_FEATURES_KEY
        :param scale: Whether to scale the data, defaults to True
        :param compute_pca: Whether to compute PCA, defaults to False
        :param pca_key_added: Key for the PCA results, defaults to 'X_pca'
        :param compute_umap: Whether to compute UMAP, defaults to False
        :param umap_key_added: Key for the UMAP results, defaults to 'X_umap'
        :param compute_tsne: Whether to compute t-SNE, defaults to False
        :param tsne_key_added: Key for the t-SNE results, defaults to 'X_tsne'
        :raises RuntimeError: If any of the operations fail
        :return: A dictionary containing the operations log
        """

        log = []

        wsi, log = self._load_wsi_with_checks(log)

        # Prerequisite: Check tissue segmentation exists
        log = self._check_tissue_key(wsi, tissue_key, log)

        # Prerequisite: Check tissue tiles exist
        log = self._check_tile_key(wsi, tile_key, log)

        # Prerequisite: Check patch features exist
        log = self._check_patch_features_key(wsi, patch_features_key, log)

        # if you have made it here that means checks passed and you can proceed with the reduction
        adata = wsi[patch_features_key]
        log.append(f"\n Successfully accessed patch features with key {patch_features_key}.")

        # reduce the adata - error handling done internally
        log, adata = reduce_data(
            adata=adata,  # type: ignore
            wsi_name=self.wsi_name,
            log=log,
            scale=scale,
            compute_neighbors=compute_neighbors,
            neighbors_key_added=neighbors_key_added,
            compute_pca=compute_pca,
            pca_key_added=pca_key_added,
            compute_umap=compute_umap,
            umap_key_added=umap_key_added,
            compute_tsne=compute_tsne,
            tsne_key_added=tsne_key_added,
        )

        try:
            _write_wsi_to_zarr_store(wsi, self.wsi_store_path)
            wsi.close()
            log.append(f"\n WSI written to zarr store at {self.wsi_store_path}.")

            asset_dict = {
                "operations_log": print_log(log),
                "path_to_zarr": str(self.wsi_store_path),
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to write WSI to zarr store. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        return asset_dict

    def run_leiden_clustering(
        self,
        tissue_key: str = TISSUES_KEY,
        tile_key: str = TILES_KEY,
        patch_features_key: str = PATCH_FEATURES_KEY,
        leiden_resolution: float = 0.75,
        leiden_key: str | None = None,
        neighbors_key_added: str = 'neighbors',
    ) -> dict:
        """
        Runs leiden clustering after performing all necessary processing checks
        """

        log = []

        wsi, log = self._load_wsi_with_checks(log)

        # Prerequisite: Check tissue segmentation exists
        log = self._check_tissue_key(wsi, tissue_key, log)

        # Prerequisite: Check tissue tiles exist
        log = self._check_tile_key(wsi, tile_key, log)

        # Prerequisite: Check patch features exist
        log = self._check_patch_features_key(wsi, patch_features_key, log)

        # if you have made it here that means checks passed and you can proceed with the reduction
        adata = wsi[patch_features_key]
        log.append(f"\n Successfully accessed patch features with key {patch_features_key}.")

        # error handling done internally
        leiden_key = leiden_key or f'leiden_{leiden_resolution}'
        adata = run_leiden_clustering(
            adata=adata,  # type: ignore
            leiden_key=leiden_key,
            leiden_resolution=leiden_resolution,
            wsi_name=self.wsi_name,
            log=log,
            neighbors_key_added=neighbors_key_added,
        )
        n_leiden_cluster = adata.obs[leiden_key].nunique()
        leiden_clusters = list(adata.obs[leiden_key].unique())

        try:
            _write_wsi_to_zarr_store(wsi, self.wsi_store_path)
            wsi.close()
            log.append(f"\n WSI written to zarr store at {self.wsi_store_path}.")

            asset_dict = {
                'leiden_key': leiden_key,
                "operations_log": print_log(log),
                "n_leiden_clusters": n_leiden_cluster,
                "leiden_clusters": leiden_clusters,
                "path_to_zarr": str(self.wsi_store_path),
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to write WSI to zarr store. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        return asset_dict

    def visualize_morphological_clusters_on_wsi(
        self,
        tissue_key: str = TISSUES_KEY,
        tile_key: str = TILES_KEY,
        tissue_id: int | None = None,
        patch_features_key: str = PATCH_FEATURES_KEY,
        leiden_resolution: float = 0.75,
        leiden_key: str | None = None,
        dpi: int = 200,
        figure_title: str | None = None,
        fig_size: tuple[int, int] = (16, 8),
        cluster_labels_key: dict | None = None,
    ) -> dict[str, str]:
        """
        Visualize feature space and spatial distribution of clusters in a WSI using UMAP and Leiden clustering.

        Generates a two-panel visualization:
            - Left: UMAP of patch features colored by cluster.
            - Right: WSI with tiles colored by cluster.

        cluster_labels_key is a dictionary mapping cluster labels to human-readable names. If provided then the cluster labels in legend will be set to these instead of the numerical labels.

        Args:
            tissue_key (str, optional): Key for tissue segmentation. Defaults to "tissues".
            tile_key (str, optional): Key for tile extraction. Defaults to "tissue_tiles".
            patch_features_key (str, optional): Key for patch features AnnData. Defaults to "uni_tissue_tiles_1024".
            leiden_resolution (float, optional): Leiden clustering resolution. Defaults to 0.75.
            dpi (int, optional): Figure DPI. Defaults to 200.
            title (Optional[str], optional): Title for the figure. Defaults to None.
            fig_size (Tuple[int, int], optional): Figure size (in inches). Defaults to (16, 8).
            tissue_id (Optional[int], optional): If set, visualize only this tissue region.

        Returns:
            dict: Contains "visualization_save_path" (str): Path to the saved figure.
        """

        log = []

        visualization_save_path = self.wsi_job_dir / f"{self.wsi_name}_morphological_clusters.png"

        wsi, log = self._load_wsi_with_checks(log)

        # Prerequisite: Check tissue segmentation exists
        log = self._check_tissue_key(wsi, tissue_key, log)

        # Prerequisite: Check if tissue_id is valid
        log = self._check_valid_tissue_id(wsi, tissue_id, tissue_key, log)

        # Prerequisite: Check tissue tiles exist
        log = self._check_tile_key(wsi, tile_key, log)

        # Prerequisite: Check patch features exist
        log = self._check_patch_features_key(wsi, patch_features_key, log)

        # Load the adata object with the features
        adata = wsi[patch_features_key]
        assert isinstance(adata, sc.AnnData), f"Expected adata to be sc.AnnData, got {type(adata)}"
        log.append(f"\n Successfully accesssed patch features. Using patch_features_key: {patch_features_key}.")

        # check if leiden clustering has been run already. Need it for visualization
        leiden_key = leiden_key or f'leiden_{leiden_resolution}'
        log = self._check_clustering_key(adata, leiden_key, log)

        fig, axes = plt.subplots(1, 2, figsize=fig_size, dpi=dpi)

        # plot the WSI without heatmap on axes[0]
        try:
            zs.pl.tissue(  # type: ignore
                wsi,
                ax=axes[0],
                tissue_id=tissue_id,
                tissue_key=tissue_key,
                show_contours=False,
                show_id=False,
                mark_origin=True,
                scalebar=True,
            )
            axes[0].set_title("Whole slide image")
            log.append("\n Plotted WSI without heatmap on the left panel.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to plot WSI for '{self.wsi_name}'. "
                f"Ensure the WSI object contains the necessary data. Original error: {e}"
                f"Operations log: {print_log(log)}"
            ) from e

        # plot on ax[1] the WSI with clusters
        try:
            # the keys in the palette must match the keys in the color column of adata.obs
            # keys in the palette are used as the labels in the legend

            if cluster_labels_key is not None:
                # Ensure all cluster labels are present in cluster_labels_key
                cluster_values = adata.obs[leiden_key].unique()
                missing_labels = set(cluster_values) - set(cluster_labels_key.keys())
                assert not missing_labels, (
                    f"cluster_labels_key is missing labels: {missing_labels}. "
                    f"All cluster labels in adata.obs['{leiden_key}'] must be present in cluster_labels_key."
                    f"All clusters in adata.obs['{leiden_key}']: {cluster_values}"
                )

                mapped_col = f'{leiden_key}_mapped'
                adata.obs[mapped_col] = adata.obs[leiden_key].map(cluster_labels_key)

                num_clusters = len(adata.obs[mapped_col].unique())
                palette = sns.color_palette("tab20", num_clusters).as_hex()
                palette_map = {cat: color for cat, color in zip(adata.obs[mapped_col].unique(), palette)}

            else:
                num_clusters = adata.obs[leiden_key].nunique()
                palette = sns.color_palette("tab20", num_clusters).as_hex()
                palette_map = {cat: color for cat, color in zip(adata.obs[leiden_key].unique(), palette)}

            zs.pl.tiles(  # type: ignore
                wsi,
                feature_key=patch_features_key,
                tile_key=tile_key,
                color=leiden_key,
                alpha=0.5,
                palette=palette_map,
                show_contours=False,
                show_id=False,
                ax=axes[1],
                tissue_id=tissue_id,
                mark_origin=True,
                scalebar=True,
            )
            axes[1].set_title("Morphological neighborhoods")
            log.append("\n Plotted morphological features on WSI tiles.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to plot morphological features for WSI '{self.wsi_name}'. "
                f"Ensure the WSI object contains the necessary data. Original error: {e}"
                f"Operations log: {print_log(log)}"
            ) from e

        if figure_title is None:
            figure_title = f"Feature Space Visualization for {self.wsi_name} (Leiden resolution: {leiden_resolution})"

        # TODO: title position arg
        fig.suptitle(figure_title, y=1.05)
        fig.tight_layout()
        plt.close(fig)

        # Save the figure
        try:
            fig.savefig(visualization_save_path, bbox_inches="tight")
            log.append(f"\n Saved visualization image to {visualization_save_path}.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to save visualization image to '{visualization_save_path}'. Original error: {e}"
                f"Operations log: {print_log(log)}"
            ) from e

        try:
            _write_wsi_to_zarr_store(wsi, self.wsi_store_path)
            wsi.close()
            log.append(f"\n WSI written to zarr store at {self.wsi_store_path}.")

            asset_dict = {"visualization_save_path": str(visualization_save_path), "operations_log": print_log(log)}

        except Exception as e:
            raise RuntimeError(
                f"Failed to write WSI to zarr store. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        return asset_dict

    def get_topk_close_patch_coords_to_clusters(
        self,
        tile_key: str = TILES_KEY,
        patch_features_key: str = PATCH_FEATURES_KEY,
        leiden_resolution: float = 0.75,
        leiden_key: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Identify and return the coordinates of the top-k patches closest to each Leiden cluster centroid.

        Args:
            tile_key (str, optional): Key for tile extraction table. Defaults to "tissue_tiles".
            patch_features_key (str, optional): Key for patch features AnnData. Defaults to "uni_tissue_tiles_1024".
            leiden_resolution (float, optional): Leiden clustering resolution. Defaults to 0.75.
            top_k (int, optional): Number of patches per cluster to select. Defaults to 5.

        Returns:
            dict: Contains:
            - "closest_patches_coords": Mapping of cluster labels to patch coordinate dicts.
            - "top_k_save_path": Path to the JSON file with closest patch info.
            - "top_k": Number of patches per cluster.
            - "leiden_resolution": The resolution parameter used.
        """

        log = []

        top_k_save_path = self.wsi_job_dir / f"{self.wsi_name}_{top_k}top_k_closest_to_clusters.json"

        wsi, log = self._load_wsi_with_checks(log)

        # Prerequisite: Check tissue tiles exist
        log = self._check_tile_key(wsi, tile_key, log)

        # Prerequisite: Check patch features exist
        log = self._check_patch_features_key(wsi, patch_features_key, log)

        # Check if the required keys exist in adata
        closest_patches_coords = {}
        adata = wsi[patch_features_key]
        X = np.array(adata.X)
        assert isinstance(adata, sc.AnnData), f"Expected adata to be sc.AnnData, got {type(adata)}"
        log.append(
            f"\n Successfully accessed patch features with key {patch_features_key}, and accessed adata.X to get features"
        )

        leiden_key = leiden_key or f"leiden_{leiden_resolution}"
        try:
            labels = np.array(adata.obs[leiden_key].astype(str).values)
            unique_labels = np.unique(labels)
            log.append(f"\n Found {len(unique_labels)} unique clusters in the WSI.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to access '{leiden_key}' in adata.obs for WSI '{self.wsi_name}'.\n"
                f"Available keys in adata.obs: {list(adata.obs.keys())}.\n"
                f"Original error: {e}\nOperations log: {print_log(log)}"
            ) from e

        try:
            tile_ids = adata.obs["tile_id"].values
        except Exception as e:
            raise RuntimeError(
                f"Failed to access 'tile_id' in adata.obs for WSI '{self.wsi_name}'.\n"
                f"Available keys in adata.obs: {list(adata.obs.keys())}.\n"
                f"Original error: {e}\nOperations log: {print_log(log)}"
            ) from e

        try:
            # Compute centroids for each cluster
            centers = np.vstack([X[labels == c].mean(axis=0) for c in unique_labels])

            for i, c in enumerate(unique_labels):
                idxs = np.where(labels == c)[0]
                dists = np.linalg.norm(X[idxs] - centers[i], axis=1)
                top_idx_in_cluster = idxs[np.argsort(dists)[:top_k]]

                # List of tile ids in the cluster
                tile_ids_in_cluster = tile_ids[top_idx_in_cluster].tolist()

                # List of coords dicts (top_x, top_y, base_height, base_width, level)
                tile_coords_in_cluster = patch_idx_to_patch_coord(
                    wsi=wsi,
                    list_of_patch_ids=tile_ids_in_cluster,
                    tile_key=tile_key,
                    patch_features_key=patch_features_key,
                )

                # Save for cluster c
                closest_patches_coords[f"cluster_{c}"] = tile_coords_in_cluster

                log.append(f"Computed closest patches for cluster {c}.")

        except Exception as e:
            raise RuntimeError(
                f"Failed to compute closest patches for clusters in WSI '{self.wsi_name}'. "
                f"Ensure the adata object contains the necessary data. Original error: {e}"
                f" Operations log: {print_log(log)}"
            ) from e

        try:
            with open(top_k_save_path, 'w') as f:
                json.dump(closest_patches_coords, f)
        except Exception as e:
            raise RuntimeError(f"Failed to write closest patches to '{top_k_save_path}'. Original error: {e}") from e

        try:
            _write_wsi_to_zarr_store(wsi, self.wsi_store_path)
            wsi.close()
            log.append(f"\n WSI written to zarr store at {self.wsi_store_path}.")

            asset_dict = {
                "closest_patches_coords": closest_patches_coords,
                "top_k_save_path": str(top_k_save_path),
                "top_k": top_k,
                "leiden_resolution": leiden_resolution,
                "operations_log": print_log(log),
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to write WSI to zarr store. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        return asset_dict

    def read_rectangle_region_from_wsi(
        self,
        region: dict,
        custom_save_dir: str | None = None,
    ) -> dict:
        """
        Extract a rectangular region from a Whole Slide Image (WSI) and save it as a JPEG.

        Args:
            region (dict): Specifies the region with keys:
            - 'top_left_x' (int): X-coordinate of the top-left corner.
            - 'top_left_y' (int): Y-coordinate of the top-left corner.
            - 'base_width' (int): Width of the region in pixels.
            - 'base_height' (int): Height of the region in pixels.
            - 'base_level' (int): Pyramid level for extraction.
            custom_save_dir (Optional[str], optional): Directory to save the image. Defaults to the WSI job directory.

        Returns:
            dict: Contains "region_save_path" (str): Path to the saved image.
        """

        log = []

        wsi, log = self._load_wsi_with_checks(log)

        # Check that all required keys are present
        required_keys = ['top_left_x', 'top_left_y', 'base_width', 'base_height', 'base_level']
        for key in required_keys:
            if key not in region:
                raise ValueError(
                    f"Region dictionary must contain the key '{key}'. Provided keys: {list(region.keys())}"
                    f"Operations log: {print_log(log)}"
                )
            log.append(f"\n Found required key '{key}' in region dictionary.")

        # Use custom_save_dir if provided, else use wsi_job_dir
        save_dir = Path(custom_save_dir) if custom_save_dir else self.wsi_job_dir
        region_save_path = (
            save_dir
            / f"{self.wsi_name}_region_x{region['top_left_x']}_y{region['top_left_y']}_H{region['base_height']}_W{region['base_width']}.png"
        )
        log.append(f"\n Region will be saved to {region_save_path}.")

        # Read the region
        try:
            region_arr = wsi.read_region(
                x=region['top_left_x'],
                y=region['top_left_y'],
                width=region['base_width'],
                height=region['base_height'],
                level=region['base_level'],
            )
            log.append("\n Successfully read the specified region from WSI.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to read region from WSI '{self.wsi_name}'. Original error: {e}Operations log: {print_log(log)}"
            ) from e

        try:
            # Convert array region to PIL Image and save
            region_img = Image.fromarray(region_arr)
            region_img.save(region_save_path)
            log.append(f"\n Saved the region image to {region_save_path}.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to save region image to '{region_save_path}'. "
                f"Original error: {e}"
                f"Operations log: {print_log(log)}"
            ) from e

        try:
            _write_wsi_to_zarr_store(wsi, self.wsi_store_path)
            wsi.close()
            log.append(f"\n WSI written to zarr store at {self.wsi_store_path}.")

            asset_dict = {
                "region_save_path": str(region_save_path),
                "operations_log": print_log(log),
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to write WSI to zarr store. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        return asset_dict

from smolagents import tool

from nova.toolkits.lazyslide.toolkit_base import PATCH_FEATURES_KEY, TILES_KEY, TISSUES_KEY
from nova.toolkits.lazyslide.toolkit_visualization import LazySlideVisualizationToolKit


@tool
def visualize_wsi_tool(
    wsi_path: str,
    job_dir: str,
    load_from_src: bool,
    add_contours: bool = False,
    add_tiles: bool = False,
    tissue_key: str = TISSUES_KEY,
    tile_key: str = TILES_KEY,
    tissue_id: int | None = None,
    title: str | None = None,
    fig_size: tuple[int, int] = (8, 8),
    dpi: int = 200,
    color_tiles_by: str | None = None,
    patch_features_key: str | None = None,
) -> dict:
    """
    Visualize the WSI at the provided path and optionally adding tissue contours and tile boundaries.
    If `color_tiles_by` is provided, tiles will be colored by the specified key in the wsi[patch_features_key].obs dataframe.

    Notes on using this tool:
        - Load the WSI from the specified path with necessary checks.
        - Optionally is initialized with a pre-existing Zarr store if `load_from_src` is True.
        - By default visualized the WSI
        - If `add_contours` is True, adds tissue segmentation contours to the visualization.
        - If `add_tiles` is True, overlays tile boundaries on the visualization.
        - If `tissue_id` is specified, only shows the given tissue ID. None will show all tissues.
        - If `color_tiles_by` is provided, then tiles will be colored by the specified key in the wsi[patch_features_key].obs dataframe.
            MUST provide `patch_features_key` if `color_tiles_by` is specified.
        - Saves the visualization as a PNG image at visualization_save_path (see returns).
        - Uses the lazyslide backend to interact with the WSI.

    Prerequisites to run this tool:
        - Tissue segmentation must be run before any visualization.
        - If loading from source, tissues might be present in the zarr file.
        - To add contours, tissue segmentation must be run
        - Tile extraction must be run before visualization if adding tile boundaries.
        - If `color_tiles_by` is specified, the `patch_features_key` must be provided and must be present in the zarr file.
        - If `color_tiles_by` is specified, the key must be present in the wsi[patch_features_key].obs dataframe.
        - Uses the lazyslide backend to interact with the WSI.

    The tool returns:
        dict: Dictionary containing:
            - 'visualization_save_path': absolute path to the saved visualize in PNG image.
            - 'operations_log': Operations performed during visualization.

    Args:
        wsi_path: Path to the input WSI file.
        job_dir: Directory where outputs and visualization image will be saved.
        load_from_src: If True, attempts to load the WSI from a
            pre-existing Zarr store.
        add_contours: If True, add tissue segmentation contours to the visualization.
            Defaults to False.
        add_tiles: If True, overlay tile boundaries on the visualization.
            Defaults to False.
        tissue_key: Key under which tissue segmentation results are stored.
            Defaults to "tissues".
        tile_key: Key under which tile extraction results are stored.
            Defaults to "tissue_tiles".The format to use is typically is tiles_px{tile_px}_mpp{mpp}_overlap{overlap}.
        tissue_id: If specified, only show the given tissue ID.
            Defaults to None (all tissues are shown).
        title: Custom title for the visualization. If None, an automatic
            title is generated. Defaults to None.
        fig_size: Figure size in inches (width, height). Defaults to (8, 8).
        dpi: Dots per inch for the output image. Defaults to 200.
        color_tiles_by: If provided, tiles will be colored by the specified key in the wsi[patch_features_key].obs dataframe.
            Must be a key in the wsi[patch_features_key].obs dataframe.
        patch_features_key: Key under which patch features AnnData is stored. Defaults to None.

    """

    viz_tool_kit = LazySlideVisualizationToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=load_from_src)
    return viz_tool_kit.visualize_wsi(
        add_contours=add_contours,
        add_tiles=add_tiles,
        tissue_key=tissue_key,
        tile_key=tile_key,
        tissue_id=tissue_id,
        title=title,
        fig_size=fig_size,
        dpi=dpi,
        color_tiles_by=color_tiles_by,
        patch_features_key=patch_features_key,
    )


@tool
def reduce_single_wsi_patch_feature_space_tool(
    wsi_path: str,
    job_dir: str,
    load_from_src: bool = True,
    tissue_key: str = TISSUES_KEY,
    tile_key: str = TILES_KEY,
    patch_features_key: str = PATCH_FEATURES_KEY,
    scale_data: bool = True,
    compute_neighbors: bool = True,
    neighbors_key_added: str = 'neighbors',
    compute_pca: bool = True,
    pca_key_added: str = 'X_pca',
    compute_umap: bool = False,
    umap_key_added: str = 'X_umap',
    compute_tsne: bool = False,
    tsne_key_added: str = 'X_tsne',
) -> dict:
    """
    Reduce the patch feature space of the WSI at the provided path using PCA, UMAP, neighbors, and/ or t-SNE. Results stored in the AnnData object in zarr file for the WSI.

    Must be called before clustering and visualization of the patch feature space. Clustering needs the neighbors graph to be computed from this function.

    Notes on using this tool:
        - Load the WSI from the specified path with necessary checks.
        - Optionally is initialized with a pre-existing Zarr store if `load_from_src` is True.
        - Scales the patch features data if `scale_data` is True.
        - Computes neighbors graph if `compute_neighbors` is True or if UMAP, or t-SNE is being computed.
            - wsi[patch_features_key].uns[neighbors_key_added] dict contains {'connectivities_key', 'distances_key',  'params': {'method', 'metric', 'n_neighbors', 'random_state'}}
            - wsi[patch_features_key].obsp[neighbors_key_added] contains `{neighbors_key_added}_connectivities` (0-1 indicating strength of connection between patches) and `{neighbors_key_added}_distances` (distance between patches).
        - Computes PCA if `compute_pca` is True.
            - PCA results stored in wsi[patch_features_key].obsm[pca_key_added]
            - wsi[patch_features_key].uns[pca_key_added] dict contains `variance` and `variance_ratio`, which is percent variance explained by each component.
        - Computes UMAP if `compute_umap` is True.
            - UMAP results stored in wsi[patch_features_key].obsm[umap_key_added]
            - wsi[patch_features_key].uns[umap_key_added] dict contains `params` dict with UMAP parameters used for computation.
        - Computes t-SNE if `compute_tsne` is True.
            -  t-SNE results stored in wsi[patch_features_key].obsm[tsne_key_added]
            - wsi[patch_features_key].uns[tsne_key_added] contains `params` dict with t-SNE parameters used for computation.
        - wsi[patch_features_key].obsm[{reduction_method}_key_added] contains the reduced feature space data to 2 dimensions.
        - Uses scanpy for data reduction.

    Prerequisites to run this tool:
        - Tissue segmentation, tiling, and patch feature extraction must be run before reduction as those steps are required to generate the patch features space

    The tool returns:
        dict: Dictionary containing:
            - 'path_to_zarr': path to the zarr file containing the WSI data and the results of the reduction.
            - 'operations_log': Log of operations performed during reduction.

    Args:
        wsi_path: Path to the WSI file.
        job_dir: Directory to save outputs and intermediate files.
        load_from_src: Whether to load from an existing Zarr store. Defaults to True.
        tissue_key: Key for tissue segmentation. Defaults to "tissues".
        tile_key: Key for tile extraction. Defaults to "tissue_tiles".
        patch_features_key: Key for patch features AnnData. Defaults to "uni_tissue_tiles".
        scale_data: Whether to scale the patch features data before reduction. Defaults to True.
        compute_pca: Whether to compute PCA on the patch features data. Defaults to True.
        compute_neighbors: Whether to compute neighbors graph for the patch features data. Defaults to True.
        neighbors_key_added: Key under which neighbors graph will be stored in the AnnData object. Defaults to 'neighbors'.
        pca_key_added: Key under which PCA results will be stored in the AnnData object. Defaults to 'X_pca'.
        compute_umap: Whether to compute UMAP on the patch features data. Defaults to False.
        umap_key_added: Key under which UMAP results will be stored in the AnnData object. Defaults to 'X_umap'.
        compute_tsne: Whether to compute t-SNE on the patch features data. Defaults to False.
        tsne_key_added: Key under which t-SNE results will be stored in the AnnData object. Defaults to 'X_tsne'.
    """

    viz_tool_kit = LazySlideVisualizationToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=load_from_src)
    return viz_tool_kit.feature_space_reduction(
        tissue_key=tissue_key,
        tile_key=tile_key,
        patch_features_key=patch_features_key,
        scale=scale_data,
        compute_neighbors=compute_neighbors,
        neighbors_key_added=neighbors_key_added,
        compute_pca=compute_pca,
        pca_key_added=pca_key_added,
        compute_umap=compute_umap,
        umap_key_added=umap_key_added,
        compute_tsne=compute_tsne,
        tsne_key_added=tsne_key_added,
    )


@tool
def run_leiden_clustering_tool(
    wsi_path: str,
    job_dir: str,
    load_from_src: bool = True,
    tissue_key: str = TISSUES_KEY,
    tile_key: str = TILES_KEY,
    patch_features_key: str = PATCH_FEATURES_KEY,
    leiden_resolution: float = 0.75,
    leiden_key: str | None = None,
    neighbors_key_added: str = 'neighbors',
) -> dict:
    """
    Run Leiden clustering on the patch feature space of the WSI at the provided path. Stores results in the AnnData object in zarr file for the WSI.

    Notes on using this tool:
        - Load the WSI from the specified path with necessary checks.
        - Optionally is initialized with a pre-existing Zarr store if `load_from_src` is True.
        - Runs Leiden clustering on the patch features. Stores results under `leiden_key` if provided, otherwise defaults to 'leiden_{leiden_resolution}'.
        - Results are stored under wsi[patch_features_key].obs[leiden_key] as a pd.Series. The values are cluster number as strings (ex: '0', '1', '2', ...). This is categorical data.
        - wsi[patch_features_key].obs has columns `tile_id`, `library_id` (this is the `tile_key`), and after successful clustering, `leiden_key` or `leiden_{leiden_resolution}` as default.
        - wsi[patch_features_key].uns[leiden_key] contains `params` dict with Leiden parameters used for clustering.
        - wsi[patch_features_key].uns['{leiden_key}_colors'] contains a list of colors for each cluster.

    Prerequisites to run this tool:
        - Tissue segmentation, tiling, and patch feature extraction must be run before clustering.
        - It is recommended to run feature space reduction (PCA, UMAP, etc.) before clustering, but it is not strictly required.
        - If loading from source, tissues, tiles, and patch features might be present in the zarr file.
        - Needs the neighbors graph to be computed from the feature space reduction step.
        - Uses the lazyslide backend to interact with the WSI. Uses scanpy for clustering.

    The tool returns:
        dict: Dictionary containing:
            - 'leiden_key': Key under which Leiden clustering results are stored in the AnnData object.
            - 'n_leiden_clusters': Number of Leiden clusters found.
            - 'leiden_clusters': List of unique Leiden cluster labels as strings.
            - 'path_to_zarr': path to the zarr file containing the WSI data and the results of the clustering.
            - 'operations_log': Log of operations performed during clustering.

    Args:
        wsi_path: Path to the WSI file.
        job_dir: Directory to save outputs and intermediate files.
        load_from_src: Whether to load from an existing Zarr store. Defaults to True.
        tissue_key: Key for tissue segmentation. Defaults to "tissues".
        tile_key: Key for tile extraction. Defaults to "tissue_tiles".
        patch_features_key: Key for patch features AnnData. Defaults to "uni_tissue_tiles_1024".
        leiden_resolution: Resolution parameter for Leiden clustering. Defaults to 0.75.
        leiden_key: Key under which Leiden clustering results will be stored in the AnnData object. If None, defaults to 'leiden_{leiden_resolution}'.
        neighbors_key_added: Key under which neighbors graph will be stored in the AnnData object. Defaults to 'neighbors'.
    """
    viz_tool_kit = LazySlideVisualizationToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=load_from_src)
    return viz_tool_kit.run_leiden_clustering(
        tissue_key=tissue_key,
        tile_key=tile_key,
        patch_features_key=patch_features_key,
        leiden_resolution=leiden_resolution,
        leiden_key=leiden_key,
        neighbors_key_added=neighbors_key_added,
    )


@tool
def visualize_morphological_clusters_on_wsi_tool(
    wsi_path: str,
    job_dir: str,
    load_from_src: bool = True,
    tissue_key: str = TISSUES_KEY,
    tile_key: str = TILES_KEY,
    tissue_id: int | None = None,
    patch_features_key: str = PATCH_FEATURES_KEY,
    leiden_resolution: float = 0.75,
    cluster_labels_key: dict | None = None,
    dpi: int = 200,
    figure_title: str | None = None,
    fig_size: tuple[int, int] = (16, 8),
) -> dict:
    """
    Make a subplot with 1 row and 2 columns. First column is the UMAP of the patch features space. Second column is the WSI with tiles colored by cluster.

    Can provide optional cluster `cluster_labels_key` to label the clusters on the plot, otherwise default labels used.

    Notes on using this tool:
        - Load the WSI from the specified path with necessary checks.
        - Optionally is initialized with a pre-existing Zarr store if `load_from_src` is True.
        - Run UMAP and Leiden clustering on the patch features.
        - Make a (1, 2) panel figure that will show the UMAP and clusters on the left and the WSI with tiles colored by cluster on the right.
        - `patch_features_key` is generally set to "{model}_{tile_key}"
        - if valid `cluster_labels_key` is provided, it will be label the clusters on the plot.
        - Save the figure to the job directory as a PNG image with the name "{wsi_name}_morphological_clusters.png".

    Prerequisites to run this tool:
        - Tissue segmentation must be run before any visualization.
        - Need to run tiling and patch feature extraction before visualization.
        - If loading from source, tissues, tiles, and patch features might be present in the zarr file.
        - Reduction of feature space must be performed beforehand.
        - Clustering of the patch features must be performed beforehand.
        - Uses the lazyslide backend to interact with the WSI.

    The tool returns:
        dict: Dictionary containing:
            - 'visualization_save_path': absolute path to the saved visualize in PNG image.
            - 'operations_log': Operations performed during visualization.

    Args:
        wsi_path: Path to the WSI file.
        job_dir: Directory to save outputs.
        load_from_src (bool, optional): Whether to load from a zarr store. Defaults to False.
        tissue_key: Key for tissue segmentation. Defaults to "tissues".
        tile_key: Key for tile extraction. Defaults to "tissue_tiles".
        tissue_id: If set, visualize only this tissue region.
        patch_features_key: Key for patch features AnnData. Defaults to "uni_tissue_tiles_1024".
        leiden_resolution: Leiden clustering resolution. Defaults to 0.75.
        dpi: Figure DPI. Defaults to 200.
        figure_title: Title for the figure. If None, generates a default.
        fig_size: Figure size (inches). Defaults to (16, 8).
        cluster_labels_key: Optional dict of keys for cluster labels to be used in the visualization. Keys must be str(int) format and must match the number of leiden clusters.
    """
    viz_tool_kit = LazySlideVisualizationToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=load_from_src)
    return viz_tool_kit.visualize_morphological_clusters_on_wsi(
        tissue_key=tissue_key,
        tile_key=tile_key,
        tissue_id=tissue_id,
        patch_features_key=patch_features_key,
        leiden_resolution=leiden_resolution,
        dpi=dpi,
        figure_title=figure_title,
        fig_size=fig_size,
        cluster_labels_key=cluster_labels_key,
    )


@tool
def get_topk_close_patch_coords_to_embedding_space_clusters_tool(
    wsi_path: str,
    job_dir: str,
    load_from_src: bool = True,
    tile_key: str = TISSUES_KEY,
    patch_features_key: str = PATCH_FEATURES_KEY,
    leiden_resolution: float = 0.75,
    top_k: int = 5,
) -> dict:
    """

    Finds the top-k patches closest to each cluster center in the WSI's patch feature space. Cluster center is defined as the mean of the patch features in the original space for each cluster.

    JSON file with patches has structure:
        {"cluster_0": [{"patch_id": int, "base_height": int, "base_width": int, "base_level": int, "top_left_x": int, "top_left_y": int}, ...], ..., "cluster_i": [...]}
    cluster ids run from 0 to number of clusters - 1.
    Coordinates are with respect to the base level
    closest_patches_coords stores this information in a dictionary format.

    Notes on using this tool:
        - Computes cluster centroids as the mean of patch features for each cluster.
        - For each cluster, finds the top-k patches closest to the centroid based on Euclidean distance.
        - Saves the closest patch coordinates and cluster information to a JSON file.
        - `patch_features_key` is generally set to "{model}_{tile_key}"

    Prerequisites:
        - visualize_single_wsi_feature_space_tool must have been run to generate clusters
        - Tissue segmentation, tiling, patch feature extraction, and Leiden clustering (visualize_single_wsi_feature_space_tool) must have been run.
        - If loading from source, tissues, tiles, and patch features might be present in the zarr file.
        - Uses the lazyslide backend to interact with the WSI.

    The tool returns:
        dict: Dictionary with the following keys:
            - "closest_patches_coords": Dictionary of cluster id to list of closest patch coordinates (defined above)
            - "top_k_save_path": Path to the JSON file (closest_patches_coords) with closest patch info.
            - "top_k": Number of patches per cluster.
            - "leiden_resolution": The resolution parameter used.
            - "operations_log": Log of operations performed.

    Args:
        wsi_path: Path to the WSI file.
        job_dir: Directory to save outputs and intermediate files.
        load_from_src: Whether to load from an existing zarr store. Defaults to False.
        tile_key: Key for tile extraction table. Defaults to "tissue_tiles".
        patch_features_key: Key for patch features AnnData. Defaults to "uni_tissue_tiles_1024".
        leiden_resolution: Leiden clustering resolution. Defaults to 0.75.
        top_k: Number of patches per cluster to select. Defaults to 5.

    """
    viz_tool_kit = LazySlideVisualizationToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=load_from_src)
    return viz_tool_kit.get_topk_close_patch_coords_to_clusters(
        tile_key=tile_key,
        patch_features_key=patch_features_key,
        leiden_resolution=leiden_resolution,
        top_k=top_k,
    )


@tool
def read_rectangle_region_from_wsi_tool(
    wsi_path: str,
    job_dir: str,
    region: dict,
    load_from_src: bool = True,
    custom_save_dir: str | None = None,
) -> dict:
    """
    Given the region, reads the region from the WSI at the provided path and saves it as a PNG image.

    Notes on using this tool:
        - Loads the WSI from the specified path with necessary checks.
        - Reads the specified rectangular region from the WSI at the given pyramid level (`base_level` key in the `region` dict).
        - Saves the extracted region as a PNG image in the specified directory or the job directory.
        - file name is: "{wsi name without ext}_region_x{region['top_left_x']}_y{region['top_left_y']}_H{region['base_height']}_W{region['base_width']}.png"

    Prerequisites to run this tool:
        - The WSI must be accessible at the provided path.
        - The region coordinates and size must be valid for the WSI at the specified level.
        - All keys in the `region` dictionary must be provided (see under Args for keys).
        - Uses the lazyslide backend to interact with the WSI.

    The tool returns:
        dict: Dictionary containing:
            - "region_save_path": Path to the saved region image.
            - "operations_log": Log of operations performed during the extraction.

    Args:
        wsi_path: Path to the WSI file.
        job_dir: Directory for job outputs (used if custom_save_dir is not provided).
        region: Dictionary specifying region keys (any other keys are ignored):
            - 'top_left_x': X-coordinate of the region's top-left corner.
            - 'top_left_y': Y-coordinate of the region's top-left corner.
            - 'base_width': Width of the region (pixels at specified `base_level`).
            - 'base_height': Height of the region (pixels at specified `base_level`).
            - 'base_level': Pyramid level at which to extract the region.
        load_from_src: If True, loads from an existing zarr store. Default: True.
        custom_save_dir: Optional directory to save the region image.
            If not provided, saves to the WSI job directory under as png files.

    """
    viz_tool_kit = LazySlideVisualizationToolKit(wsi_path=wsi_path, job_dir=job_dir, load_from_src=load_from_src)
    return viz_tool_kit.read_rectangle_region_from_wsi(
        region=region,
        custom_save_dir=custom_save_dir,
    )

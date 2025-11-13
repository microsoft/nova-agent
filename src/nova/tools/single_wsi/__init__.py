from nova.tools.single_wsi.captioning import (
    caption_single_wsi_tool,
    generate_wsi_report_with_prism_tool,
    predict_wsi_label_tool,
    score_tiles_by_text_in_a_wsi_tool,
    visualize_text_prompt_similarity_on_wsi_tool,
)
from nova.tools.single_wsi.processing import (
    access_zarr_hierarchy,
    check_clustering_key_in_wsi_tool,
    check_patch_features_key_in_wsi_tool,
    check_reduction_key_in_wsi_tool,
    check_slide_features_key_in_wsi_tool,
    check_tile_key_in_wsi_tool,
    check_tissue_segmentation_key_in_wsi_tool,
    encode_wsi_tool,
    extract_patch_features_in_wsi_tool,
    extract_tissue_in_wsi_tool,
    extract_tissue_tiles_in_wsi_tool,
    read_zarr_data_tool,
    retrieve_properties_from_wsi_tool,
)
from nova.tools.single_wsi.visualization import (
    get_topk_close_patch_coords_to_embedding_space_clusters_tool,
    read_rectangle_region_from_wsi_tool,
    reduce_single_wsi_patch_feature_space_tool,
    run_leiden_clustering_tool,
    visualize_morphological_clusters_on_wsi_tool,
    visualize_wsi_tool,
)

SINGLE_wsi_TOOLS_DICT = {
    "visualize_text_prompt_similarity_on_wsi_tool": {
        "func": visualize_text_prompt_similarity_on_wsi_tool,
        "description": "Visualize text prompt similarity scores overlaid on WSI tissue regions",
    },
    "predict_wsi_label_tool": {
        "func": predict_wsi_label_tool,
        "description": "Predict WSI-level class labels using text-based zero-shot classification",
    },
    "generate_wsi_report_with_prism_tool": {
        "func": generate_wsi_report_with_prism_tool,
        "description": "Generate a pathology report for a WSI using the PRISM model",
    },
    "caption_single_wsi_tool": {
        "func": caption_single_wsi_tool,
        "description": "Generate descriptive captions for a single WSI by clustering and summarizing",
    },
    "score_tiles_by_text_in_a_wsi_tool": {
        "func": score_tiles_by_text_in_a_wsi_tool,
        "description": "Score individual tiles in a WSI based on text-based similarity criteria",
    },
    "retrieve_properties_from_wsi_tool": {
        "func": retrieve_properties_from_wsi_tool,
        "description": "Retrieve metadata and properties from a single WSI file",
    },
    "extract_tissue_in_wsi_tool": {
        "func": extract_tissue_in_wsi_tool,
        "description": "Perform tissue segmentation on a single WSI file",
    },
    "extract_tissue_tiles_in_wsi_tool": {
        "func": extract_tissue_tiles_in_wsi_tool,
        "description": "Extract tissue tiles/patches from a single WSI file",
    },
    "extract_patch_features_in_wsi_tool": {
        "func": extract_patch_features_in_wsi_tool,
        "description": "Extract patch-level features from a single WSI using foundation models",
    },
    "encode_wsi_tool": {
        "func": encode_wsi_tool,
        "description": "Encode a single WSI with slide-level features using LazySlide backend",
    },
    "check_tissue_segmentation_key_in_wsi_tool": {
        "func": check_tissue_segmentation_key_in_wsi_tool,
        "description": "Check if tissue segmentation results exist for a specific key in WSI",
    },
    "check_tile_key_in_wsi_tool": {
        "func": check_tile_key_in_wsi_tool,
        "description": "Check if tile extraction results exist for a specific key in WSI",
    },
    "check_patch_features_key_in_wsi_tool": {
        "func": check_patch_features_key_in_wsi_tool,
        "description": "Check if patch features exist for a specific key in WSI",
    },
    "check_slide_features_key_in_wsi_tool": {
        "func": check_slide_features_key_in_wsi_tool,
        "description": "Check if slide-level features exist for a specific key in WSI",
    },
    "check_clustering_key_in_wsi_tool": {
        "func": check_clustering_key_in_wsi_tool,
        "description": "Check if clustering results exist for a specific key in WSI",
    },
    "check_reduction_key_in_wsi_tool": {
        "func": check_reduction_key_in_wsi_tool,
        "description": "Check if dimensionality reduction results exist for a specific key in WSI",
    },
    "access_zarr_hierarchy": {
        "func": access_zarr_hierarchy,
        "description": "Access and explore the hierarchical structure of WSI Zarr files",
    },
    "read_zarr_data_tool": {
        "func": read_zarr_data_tool,
        "description": "Read data from a Zarr file",
    },
    "visualize_wsi_tool": {
        "func": visualize_wsi_tool,
        "description": "Create visualizations of WSI with optional tissue contours and tile overlays",
    },
    "reduce_single_wsi_patch_feature_space_tool": {
        "func": reduce_single_wsi_patch_feature_space_tool,
        "description": "Reduce dimensionality of patch features using PCA, UMAP, or t-SNE",
    },
    "run_leiden_clustering_tool": {
        "func": run_leiden_clustering_tool,
        "description": "Perform Leiden clustering on patch features for morphological analysis",
    },
    "visualize_morphological_clusters_on_wsi_tool": {
        "func": visualize_morphological_clusters_on_wsi_tool,
        "description": "Visualize morphological clusters overlaid on WSI tissue regions",
    },
    "get_topk_close_patch_coords_to_embedding_space_clusters_tool": {
        "func": get_topk_close_patch_coords_to_embedding_space_clusters_tool,
        "description": "Get coordinates of top-k patches closest to embedding space cluster centers",
    },
    "read_rectangle_region_from_wsi_tool": {
        "func": read_rectangle_region_from_wsi_tool,
        "description": "Extract rectangular regions from WSI at specified coordinates and magnification",
    },
}

# Select which tools to keep from this keys list
TOOLS_KEYS = [
    # WSI_CAPTIONING_TOOLS
    "visualize_text_prompt_similarity_on_wsi_tool",
    "predict_wsi_label_tool",
    "generate_wsi_report_with_prism_tool",
    "caption_single_wsi_tool",
    "score_tiles_by_text_in_a_wsi_tool",
    # WSI_PROCESSING_TOOLS
    "access_zarr_hierarchy",
    "check_tissue_segmentation_key_in_wsi_tool",
    "check_tile_key_in_wsi_tool",
    "check_patch_features_key_in_wsi_tool",
    "check_slide_features_key_in_wsi_tool",
    "check_clustering_key_in_wsi_tool",
    "check_reduction_key_in_wsi_tool",
    "extract_tissue_in_wsi_tool",
    "extract_tissue_tiles_in_wsi_tool",
    "extract_patch_features_in_wsi_tool",
    "encode_wsi_tool",
    "read_zarr_data_tool",
    "retrieve_properties_from_wsi_tool",
    # WSI_VISUALIZATION_TOOLS
    "visualize_wsi_tool",
    "reduce_single_wsi_patch_feature_space_tool",
    "run_leiden_clustering_tool",
    "visualize_morphological_clusters_on_wsi_tool",
    "get_topk_close_patch_coords_to_embedding_space_clusters_tool",
    "read_rectangle_region_from_wsi_tool",
]

WSI_TOOLS = {k: v for k, v in SINGLE_wsi_TOOLS_DICT.items() if k in TOOLS_KEYS}

from nova.tools.dataset_wsi.processing import (
    dataset_of_wsi_patch_coordinate_extraction_tool,
    dataset_of_wsi_patch_features_extraction_tool,
    dataset_of_wsi_slide_features_extraction_tool,
    dataset_of_wsi_tissue_segmentation_tool,
)
from nova.tools.dataset_wsi.visualization import dataset_of_wsi_create_score_heatmap_tool

DATASET_WSI_TOOLS_DICT = {
    "dataset_of_wsi_tissue_segmentation_tool": {
        "func": dataset_of_wsi_tissue_segmentation_tool,
        "description": "Perform tissue segmentation on a dataset of WSI files",
    },
    "dataset_of_wsi_patch_coordinate_extraction_tool": {
        "func": dataset_of_wsi_patch_coordinate_extraction_tool,
        "description": "Extract patch coordinates from tissue regions in a dataset of WSIs",
    },
    "dataset_of_wsi_patch_features_extraction_tool": {
        "func": dataset_of_wsi_patch_features_extraction_tool,
        "description": "Extract patch-level features from a dataset of WSIs using foundation models",
    },
    "dataset_of_wsi_slide_features_extraction_tool": {
        "func": dataset_of_wsi_slide_features_extraction_tool,
        "description": "Extract slide-level features from patch features for a dataset of WSIs",
    },
    "dataset_of_wsi_create_score_heatmap_tool": {
        "func": dataset_of_wsi_create_score_heatmap_tool,
        "description": "Create score heatmaps overlaid on WSIs for visualization and analysis",
    },
}

# Select which tools to keep from this keys list
TOOLS_KEYS = [
    # WSI_DATASET_PROCESSING_TOOLS
    "dataset_of_wsi_tissue_segmentation_tool",
    "dataset_of_wsi_patch_coordinate_extraction_tool",
    "dataset_of_wsi_patch_features_extraction_tool",
    "dataset_of_wsi_slide_features_extraction_tool",
    # WSI_DATASET_VIZ_TOOLS
    "dataset_of_wsi_create_score_heatmap_tool",
]

DATASET_WSI_TOOLS = {k: v for k, v in DATASET_WSI_TOOLS_DICT.items() if k in TOOLS_KEYS}

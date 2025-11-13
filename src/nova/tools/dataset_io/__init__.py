from nova.tools.dataset_io.wsi_dataset_io_tools import (
    dataset_of_wsi_check_patch_coordinates_exist_and_schema_tool,
    dataset_of_wsi_check_patch_features_exist_and_schema_tool,
    dataset_of_wsi_check_slide_features_exist_and_schema_tool,
    dataset_of_wsi_check_tissue_segmentation_exists_tool,
    dataset_of_wsi_get_valid_slide_paths_tool,
)

DATASET_IO_TOOLS_DICT = {
    "dataset_of_wsi_get_valid_slide_paths_tool": {
        "func": dataset_of_wsi_get_valid_slide_paths_tool,
        "description": "Get valid WSI file paths from a directory with optional extension filtering",
    },
    "dataset_of_wsi_check_tissue_segmentation_exists_tool": {
        "func": dataset_of_wsi_check_tissue_segmentation_exists_tool,
        "description": "Check if tissue segmentation files exist for a dataset of WSIs",
    },
    "dataset_of_wsi_check_patch_coordinates_exist_and_schema_tool": {
        "func": dataset_of_wsi_check_patch_coordinates_exist_and_schema_tool,
        "description": "Check if patch coordinate files exist and validate their schema for a dataset of WSIs",
    },
    "dataset_of_wsi_check_patch_features_exist_and_schema_tool": {
        "func": dataset_of_wsi_check_patch_features_exist_and_schema_tool,
        "description": "Check if patch feature files exist and validate their schema for a dataset of WSIs",
    },
    "dataset_of_wsi_check_slide_features_exist_and_schema_tool": {
        "func": dataset_of_wsi_check_slide_features_exist_and_schema_tool,
        "description": "Check if slide-level feature files exist and validate their schema for a dataset of WSIs",
    },
}

# Select which tools to keep from this keys list
TOOLS_KEYS = [
    # WSI_DATASET_IO_TOOLS
    "dataset_of_wsi_get_valid_slide_paths_tool",
    "dataset_of_wsi_check_tissue_segmentation_exists_tool",
    "dataset_of_wsi_check_patch_coordinates_exist_and_schema_tool",
    "dataset_of_wsi_check_patch_features_exist_and_schema_tool",
    "dataset_of_wsi_check_slide_features_exist_and_schema_tool",
]

WSI_DATASET_IO_TOOLS = {k: v for k, v in DATASET_IO_TOOLS_DICT.items() if k in TOOLS_KEYS}

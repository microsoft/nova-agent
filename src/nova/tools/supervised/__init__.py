from nova.tools.supervised.supervised_tools import (
    create_wsi_classification_splits,
    prepare_wsi_classification_metadata,
    train_test_wsi_classification_mil_model,
)

SUPERVISED_TOOLS_DICT = {
    "train_test_wsi_classification_mil_model": {
        "func": train_test_wsi_classification_mil_model,
        "description": "Train and test multiple instance learning models for WSI classification",
    },
    "create_wsi_classification_splits": {
        "func": create_wsi_classification_splits,
        "description": "Create train/validation/test splits for WSI classification datasets",
    },
    "prepare_wsi_classification_metadata": {
        "func": prepare_wsi_classification_metadata,
        "description": "Prepare metadata files for WSI classification experiments",
    },
}

# Select which tools to keep from this keys list
TOOLS_KEYS = [
    # SUPERVISED_TOOLS
    "train_test_wsi_classification_mil_model",
    "create_wsi_classification_splits",
    "prepare_wsi_classification_metadata",
]

SUPERVISED_TOOLS = {k: v for k, v in SUPERVISED_TOOLS_DICT.items() if k in TOOLS_KEYS}

from nova.tools.tiles.captioning import (
    caption_and_summarize_set_of_histology_images_tool,
    caption_single_histology_image_tool,
    score_single_histology_image_using_text_tool,
)
from nova.tools.tiles.processing import encode_histology_roi_tool

TILES_TOOLS_DICT = {
    "caption_single_histology_image_tool": {
        "func": caption_single_histology_image_tool,
        "description": "Generate a descriptive caption for a single histology image",
    },
    "caption_and_summarize_set_of_histology_images_tool": {
        "func": caption_and_summarize_set_of_histology_images_tool,
        "description": "Caption multiple histology images and provide a summary",
    },
    "score_single_histology_image_using_text_tool": {
        "func": score_single_histology_image_using_text_tool,
        "description": "Score a histology image based on text-based criteria",
    },
    "encode_histology_roi_tool": {
        "func": encode_histology_roi_tool,
        "description": "Encode histology region of interest into vector representation",
    },
}

# Select which tools to keep from this keys list
TOOLS_KEYS = [
    # TILES_CAPTIONING_TOOLS
    "caption_single_histology_image_tool",
    "caption_and_summarize_set_of_histology_images_tool",
    "score_single_histology_image_using_text_tool",
    # HISTOLOGY_ROI_TOOLS
    "encode_histology_roi_tool",
]

TILES_TOOLS = {k: v for k, v in TILES_TOOLS_DICT.items() if k in TOOLS_KEYS}

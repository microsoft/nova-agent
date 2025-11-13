from nova.tools.roi_nuclei_seg.contour_utils import (
    get_contour_area,
    get_contour_convex_hull,
    get_contour_perimeter,
)
from nova.tools.roi_nuclei_seg.roi_nuc_seg import (
    segment_and_classify_nuclei_in_histology_roi_tool,
)

ROI_NUCLEI_SEG_TOOLS_DICT = {
    "segment_and_classify_nuclei_in_histology_roi_tool": {
        "func": segment_and_classify_nuclei_in_histology_roi_tool,
        "description": "Segment and classify nuclei in histology ROIs into 6 classes using HoVer-Net",
    },
    "get_contour_area": {
        "func": get_contour_area,
        "description": "Calculate the area of a contour.",
    },
    "get_contour_perimeter": {
        "func": get_contour_perimeter,
        "description": "Calculate the perimeter of a contour.",
    },
    "get_contour_convex_hull": {
        "func": get_contour_convex_hull,
        "description": "Calculate the convex hull of a contour.",
    },
}

# Select which tools to keep from this keys list
TOOLS_KEYS = [
    # ROI_NUC_SEG_TOOLS
    "segment_and_classify_nuclei_in_histology_roi_tool",
    # CONTOUR_TOOLS
    "get_contour_area",
    "get_contour_perimeter",
    "get_contour_convex_hull",
]

ROI_NUCLEI_SEG_TOOLS = {k: v for k, v in ROI_NUCLEI_SEG_TOOLS_DICT.items() if k in TOOLS_KEYS}

from enum import StrEnum

from smolagents import Tool

from nova.tools.dataset_io import WSI_DATASET_IO_TOOLS
from nova.tools.dataset_wsi import DATASET_WSI_TOOLS
from nova.tools.documentation import DOCUMENTATION_TOOLS
from nova.tools.roi_nuclei_seg import ROI_NUCLEI_SEG_TOOLS
from nova.tools.single_wsi import WSI_TOOLS
from nova.tools.supervised import SUPERVISED_TOOLS
from nova.tools.tiles import TILES_TOOLS


class ToolsCategory(StrEnum):
    DATASET_IO = "dataset_io"
    DATASET_WSI = "dataset_wsi"
    DOCUMENTATION = "documentation"
    ROI_NUCLEI_SEG = "roi_nuclei_seg"
    WSI_ANALYSIS = "wsi_analysis"
    SUPERVISED = "supervised"
    TILES = "tiles"
    ALL = "all"
    NONE = "none"


ALL_TOOLS_DICT: dict[ToolsCategory, dict[str, dict[str, Tool | str]]] = {
    ToolsCategory.DATASET_IO: WSI_DATASET_IO_TOOLS,
    ToolsCategory.DATASET_WSI: DATASET_WSI_TOOLS,
    ToolsCategory.DOCUMENTATION: DOCUMENTATION_TOOLS,
    ToolsCategory.ROI_NUCLEI_SEG: ROI_NUCLEI_SEG_TOOLS,
    ToolsCategory.WSI_ANALYSIS: WSI_TOOLS,
    ToolsCategory.SUPERVISED: SUPERVISED_TOOLS,
    ToolsCategory.TILES: TILES_TOOLS,
    ToolsCategory.NONE: {},
}


def _get_tools_for_single_category(category: ToolsCategory) -> list[Tool]:
    if category == ToolsCategory.ALL:
        raise ValueError("Use get_tools_list_for_category with category=ToolsCategory.ALL to get all tools")
    tools_dict_for_category = ALL_TOOLS_DICT[category]
    tools = [tool["func"] for tool in tools_dict_for_category.values()]
    return tools  # type:ignore


def _get_tools_description_for_single_category(category: ToolsCategory) -> list[dict[str, str]]:
    if category == ToolsCategory.ALL:
        raise ValueError("Use get_tools_descriptions_for_category with category=ToolsCategory.ALL to get all tools")
    tools_dict_for_category = ALL_TOOLS_DICT[category]
    tools_descriptions = [
        {"name": tool_name, "description": tool_dict["description"]}
        for tool_name, tool_dict in tools_dict_for_category.items()
    ]
    return tools_descriptions


def get_tools_list_for_category(category: ToolsCategory) -> list[Tool]:
    if category not in ToolsCategory:
        raise ValueError(f"Invalid category: {category}, choose one of {list(ToolsCategory)}")
    if category == ToolsCategory.ALL:
        return [tool for cat in ALL_TOOLS_DICT.keys() for tool in _get_tools_for_single_category(cat)]
    return _get_tools_for_single_category(category)


def get_tools_descriptions_for_category(category: ToolsCategory) -> list[dict[str, str]]:
    if category not in ToolsCategory:
        raise ValueError(f"Invalid category: {category}, choose one of {list(ToolsCategory)}")
    if category == ToolsCategory.ALL:
        return [
            tool_desc for cat in ALL_TOOLS_DICT.keys() for tool_desc in _get_tools_description_for_single_category(cat)
        ]
    return _get_tools_description_for_single_category(category)

from smolagents import WebSearchTool

from nova.tools.documentation.documentation import (
    hovernet_docs_retriever,
    lazyslide_docs_retriever,
    trident_docs_retriever,
)

DOCUMENTATION_TOOLS_DICT = {
    "trident_docs_retriever": {
        "func": trident_docs_retriever,
        "description": "Search and retrieve information from Trident documentation for WSI processing",
    },
    "lazyslide_docs_retriever": {
        "func": lazyslide_docs_retriever,
        "description": "Search and retrieve information from LazySlide documentation for WSI analysis",
    },
    "hovernet_docs_retriever": {
        "func": hovernet_docs_retriever,
        "description": "Search and retrieve information from HoverNet documentation for nuclei segmentation and classification",
    },
    "web_search_tool": {
        "func": WebSearchTool(),
        "description": "Search the web for information to supplement answering user queries",
    },
}

# Select which tools to keep from this keys list
TOOLS_KEYS = [
    # GITHUB_DOCUMENTATION_TOOLS
    "trident_docs_retriever",
    "lazyslide_docs_retriever",
    "hovernet_docs_retriever",
    # "web_search_tool"
]

DOCUMENTATION_TOOLS = {k: v for k, v in DOCUMENTATION_TOOLS_DICT.items() if k in TOOLS_KEYS}

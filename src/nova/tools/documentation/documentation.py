from pathlib import Path

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool

from nova.paths import APIS_DIR


class GithubDocsRetrieverTool(Tool):
    """
    Tool for retrieving relevant text and code snippets from the documentation of a GitHub repository.

    This tool loads a `.txt` file containing the entire API documentation, splits it into overlapping chunks
    for better retrieval, and uses BM25 scoring to return the most relevant text segments for any natural language query.
    Useful for code agents or chatbots needing to answer questions about the GitHub API or usage patterns or solving errors arising from this API.
    """

    name = "docs_retriever_tool"
    description = """
        Retrieves the most relevant chunks from the GitHub repository documentation, including the code and readme files.
        This is useful for answering technical queries, explaining functions, or surfacing relevant examples from GitHub docs.
        It can also be helpful to answer questions about errors or features or functions while using the GitHub API or library.
        """
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The search query for the documentation retriever. "
                "Phrase your query in a clear, descriptive way; it does not need to be a full question."
            ),
        }
    }
    output_type = "string"

    def __init__(self, docs_path: str, **kwargs):
        super().__init__(**kwargs)

        # Read the local documentation file
        with open(docs_path, "r", encoding="utf-8") as f:
            doc_text = f.read()

        # Create Document objects for retrieval
        source_docs = [Document(page_content=doc_text, metadata={"source": docs_path})]

        # Split the document into smaller, overlapping chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        docs_processed = text_splitter.split_documents(source_docs)

        # Initialize BM25 retriever with the processed document chunks
        self.retriever = BM25Retriever.from_documents(docs_processed, k=10)

    def forward(self, query: str) -> str:  # type: ignore
        """Retrieve the most relevant documentation passages for a given query string."""
        assert isinstance(query, str), "The query must be a string"

        docs = self.retriever.invoke(query)
        if not docs:
            return "No relevant documentation found for your query."

        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {i} =====\n{doc.page_content}" for i, doc in enumerate(docs)]
        )


trident_docs_retriever = GithubDocsRetrieverTool(docs_path=str(Path(APIS_DIR) / "TRIDENT_API.txt"))
trident_docs_retriever.name = "trident_docs_retriever_tool"
lazyslide_docs_retriever = GithubDocsRetrieverTool(docs_path=str(Path(APIS_DIR) / "LAZYSLIDE_API.txt"))
lazyslide_docs_retriever.name = "lazyslide_docs_retriever_tool"
hovernet_docs_retriever = GithubDocsRetrieverTool(docs_path=str(Path(APIS_DIR) / "HOVERNET_API.txt"))
hovernet_docs_retriever.name = "hovernet_docs_retriever_tool"

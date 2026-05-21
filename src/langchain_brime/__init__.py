"""LangChain integration for the Brime API."""

from importlib.metadata import PackageNotFoundError, version

from langchain_brime.retrievers import BrimeRetriever
from langchain_brime.tools.extract import BrimeExtract
from langchain_brime.tools.research import BrimeResearch
from langchain_brime.tools.search import BrimeSearch

try:
    __version__ = version("langchain-brime")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "BrimeExtract",
    "BrimeResearch",
    "BrimeRetriever",
    "BrimeSearch",
    "__version__",
]

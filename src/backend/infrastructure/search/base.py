from abc import ABC, abstractmethod
from typing import List, Dict, Any

class SearchProvider(ABC):
    """
    Abstract base class for web search providers.
    Ensures all search implementations return data in a consistent format.
    """
    
    @abstractmethod
    async def search(self, query: str) -> str:
        """
        Execute a search query and return a JSON string result.
        The result should be a list of dictionaries adhering to the RawSearchResult schema:
        {
            "content": str,
            "document_name": str,
            "url": str (optional),
            "score": float (optional),
            "provider": str (optional)
        }
        """
        pass

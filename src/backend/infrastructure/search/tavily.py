import json
import logging
from .base import SearchProvider
from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)

class TavilyProvider(SearchProvider):
    """
    Search provider implementation using Tavily.
    Requires API Key. Optimized for LLM RAG.
    """
    def __init__(self, api_key: str, max_results: int = 5):
        # TavilySearch replaces the deprecated TavilySearchResults
        self.tavily = TavilySearch(
            tavily_api_key=api_key,
            max_results=max_results
        )

    async def search(self, query: str) -> str:
        try:
            # TavilySearch (langchain-tavily) returns a dict with 'results', 'images', etc.
            # Example: {'query': '...', 'results': [{'title': '...', 'url': '...', 'content': '...', 'score': 0.9}, ...]}
            raw_response = await self.tavily.ainvoke({"query": query})
            
            formatted_results = []
            
            # Check if we got a dict with 'results' key (standard Tavily API response structure via langchain wrapper)
            if isinstance(raw_response, dict) and "results" in raw_response:
                results_list = raw_response["results"]
                for res in results_list:
                    content = res.get("content", "")
                    url = res.get("url", "")
                    title = res.get("title")
                    
                    if not title:
                         title = f"Web Search Result ({url[:30]}...)"
                    
                    score = res.get("score", None) 
                    
                    formatted_results.append({
                        "content": content,
                        "document_name": title,
                        "url": url,
                        "score": score,
                        "provider": "tavily"
                    })
            # Fallback: maybe it returned a list directly (old behavior or different wrapper version)
            elif isinstance(raw_response, list):
                for res in raw_response:
                    content = res.get("content", "")
                    url = res.get("url", "")
                    title = res.get("title") or url
                    score = res.get("score", None)
                    
                    formatted_results.append({
                        "content": content,
                        "document_name": title,
                        "url": url,
                        "score": score,
                        "provider": "tavily"
                    })
            else:
                 # Unexpected format
                formatted_results.append({
                    "content": str(raw_response),
                    "document_name": "Web Search (Tavily)",
                    "url": None,
                    "provider": "tavily"
                })

            return json.dumps(formatted_results, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Tavily Search Error: {e}")
            return json.dumps([{"content": f"搜索失败: {str(e)}", "document_name": "System"}], ensure_ascii=False)

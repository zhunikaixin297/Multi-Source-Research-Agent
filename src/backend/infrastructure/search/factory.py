import json
import logging
from typing import Optional

from langchain_core.tools import tool

from .base import SearchProvider
from .duckduckgo import DuckDuckGoProvider
from .tavily import TavilyProvider
from ...core.config import settings

logger = logging.getLogger(__name__)

class SearchFactory:
    """
    Factory to create the configured Search Provider.
    """
    _instance: Optional[SearchProvider] = None

    @classmethod
    def get_provider(cls) -> SearchProvider:
        if cls._instance is None:
            config = settings.search
            provider_type = config.provider.lower()
            
            logger.info(f"Initializing Web Search Provider: {provider_type}")
            
            if provider_type == "tavily":
                if not config.api_key:
                    logger.warning("Tavily provider selected but no API Key found! Fallback to DuckDuckGo.")
                    cls._instance = DuckDuckGoProvider(max_results=config.max_results)
                else:
                    cls._instance = TavilyProvider(api_key=config.api_key, max_results=config.max_results)
            elif provider_type == "duckduckgo":
                cls._instance = DuckDuckGoProvider(max_results=config.max_results)
            else:
                logger.error(f"Unknown search provider: {provider_type}. Using default (DuckDuckGo).")
                cls._instance = DuckDuckGoProvider(max_results=config.max_results)
                
        return cls._instance

# --- Rate Limiter ---
import time

class RateLimiter:
    def __init__(self, calls_per_minute: int = 10):
        self.calls_per_minute = calls_per_minute
        self.call_timestamps = []

    def can_call(self) -> bool:
        now = time.time()
        # Clean up old timestamps
        self.call_timestamps = [ts for ts in self.call_timestamps if now - ts < 60]
        return len(self.call_timestamps) < self.calls_per_minute

    def record_call(self):
        self.call_timestamps.append(time.time())

_web_search_rate_limiter = RateLimiter(calls_per_minute=10) # slightly relaxed limit for factory usage

# --- Tool Wrapper ---

@tool
async def web_search_tool(query: str) -> str:
    """
    搜索公共互联网 (Web Search)。
    当你需要查询最新新闻、实时事件、外部公开技术文档或通用百科知识时，使用此工具。
    不要用此工具搜索公司内部机密资料。
    """
    if not _web_search_rate_limiter.can_call():
        logger.warning("Web search rate limit exceeded.")
        return json.dumps([{"content": "网络搜索被限流，请稍后再试", "document_name": "System"}], ensure_ascii=False)

    _web_search_rate_limiter.record_call()
    
    try:
        provider = SearchFactory.get_provider()
        return await provider.search(query)
    except Exception as e:
        logger.error(f"Web Search Tool Error: {e}")
        return json.dumps([{"content": f"网络搜索失败: {str(e)}", "document_name": "System"}], ensure_ascii=False)

import json
import logging
import re
from typing import Any
from .base import SearchProvider
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from ...core.config import settings

logger = logging.getLogger(__name__)

class DuckDuckGoProvider(SearchProvider):
    """
    Search provider implementation using DuckDuckGo.
    Free, no API key required.
    """
    def __init__(self, max_results: int = 5):
        # 使用配置中的 backend (默认为 auto)
        # 可选引擎包括: brave, duckduckgo, google, grokipedia, mojeek, wikipedia, yahoo, yandex
        backend = settings.search.ddg_backend
        
        logger.info(f"Initializing DuckDuckGo with backend: {backend}")
        
        wrapper = DuckDuckGoSearchAPIWrapper(
            backend=backend, 
            max_results=max_results,
            region="wt-wt",
            safesearch="moderate",
            time="y"
        )
        self.ddg = DuckDuckGoSearchResults(api_wrapper=wrapper)

    def _parse_ddg_result(self, raw_result: Any) -> list[dict]:
        """
        根据 DuckDuckGoSearchResults 的原始输出格式解析结果。
        原始格式通常为一串字符串: 'snippet: ..., title: ..., link: ..., snippet: ...'
        """
        if not raw_result:
            return []

        results = []
        
        # 1. 如果已经是结构化数据 (list/dict)，直接处理
        if isinstance(raw_result, list):
            for item in raw_result:
                if isinstance(item, dict):
                    results.append({
                        "content": item.get("snippet") or item.get("content") or str(item),
                        "document_name": item.get("title") or "DuckDuckGo Result",
                        "url": item.get("link") or item.get("href"),
                        "provider": "duckduckgo"
                    })
        elif isinstance(raw_result, dict):
            results.append({
                "content": raw_result.get("snippet") or raw_result.get("content") or str(raw_result),
                "document_name": raw_result.get("title") or "DuckDuckGo Result",
                "url": raw_result.get("link") or raw_result.get("href"),
                "provider": "duckduckgo"
            })
        
        # 2. 如果是字符串格式 (控制台显示的格式)
        else:
            text = str(raw_result).strip()
            # 移除可能存在的包裹括号 [...]
            if text.startswith('[') and text.endswith(']'):
                text = text[1:-1]
            
            # 使用正则提取 snippet, title, link 三元组
            # 这里的正则考虑到 snippet 和 title 可能包含逗号，但字段前缀是固定的
            # 更加健壮的方法是先分割不同的记录，然后再提取字段
            # 记录之间通常由 ", snippet: " 分隔
            records = re.split(r",\s+snippet:\s+", text)
            
            for i, record in enumerate(records):
                # 补全第一个记录以外的记录前缀
                if i > 0:
                    record = "snippet: " + record
                # 如果第一个记录没带前缀也补上 (虽然通常都有)
                if not record.startswith("snippet:"):
                    record = "snippet: " + record
                
                # 在单个记录中提取字段
                # 使用更加精确的正则：匹配到下一个字段前缀或字符串末尾
                s_match = re.search(r"snippet:\s*(.*?)(?:,\s*title:|$)", record, re.DOTALL)
                t_match = re.search(r"title:\s*(.*?)(?:,\s*link:|$)", record, re.DOTALL)
                l_match = re.search(r"link:\s*(https?://[^\s,\]]+)", record)
                
                if s_match and t_match and l_match:
                    results.append({
                        "content": s_match.group(1).strip(),
                        "document_name": t_match.group(1).strip(),
                        "url": l_match.group(1).strip(),
                        "provider": "duckduckgo"
                    })
                elif l_match: # 至少要有链接
                    results.append({
                        "content": s_match.group(1).strip() if s_match else record,
                        "document_name": t_match.group(1).strip() if t_match else "Web Search (DuckDuckGo)",
                        "url": l_match.group(1).strip(),
                        "provider": "duckduckgo"
                    })
        
        # 3. 兜底逻辑：如果正则没匹配到，将整块文本作为一条结果
        if not results and raw_result:
            results.append({
                "content": str(raw_result),
                "document_name": "Web Search (DuckDuckGo)",
                "url": None,
                "provider": "duckduckgo"
            })
            
        return results

    async def search(self, query: str) -> str:
        try:
            # DDG invoke is synchronous
            raw_result = self.ddg.invoke({"query": query})
            
            # 解析原始结果
            formatted_results = self._parse_ddg_result(raw_result)
            
            return json.dumps(formatted_results, ensure_ascii=False)
        except Exception as e:
            logger.error(f"DuckDuckGo Search Error: {e}")
            return json.dumps([{
                "content": f"网络搜索失败: {str(e)}", 
                "document_name": "System",
                "url": None,
                "provider": "duckduckgo"
            }], ensure_ascii=False)

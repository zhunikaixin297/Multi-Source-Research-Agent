import asyncio
import os
import sys
import json
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.backend.infrastructure.mcp_connector.tools import web_search_tool, parse_tool_output
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_web_search():
    """
    Test script to verify the Web Search tool (DuckDuckGo).
    """
    if os.getenv("RUN_WEB_SEARCH_TESTS") != "1":
        pytest.skip("RUN_WEB_SEARCH_TESTS not set, skipping web search test")
    test_queries = [
        "2024年巴黎奥运会金牌榜",
        "DeepSeek-V3 模型特点",
        "Python 3.13 新特性"
    ]

    for query in test_queries:
        logger.info(f"Testing web search for: '{query}'")
        try:
            # 1. Invoke the tool
            result_json = await web_search_tool.ainvoke({"query": query})
            
            # 2. Parse the result
            results = parse_tool_output(result_json)
            
            if results and results[0]["content"] != "网络搜索失败":
                logger.info(f"Successfully received search results for '{query}':")
                for res in results:
                    # Print more details if available
                    source = res['document_name']
                    if res.get('url'):
                        source += f" ({res['url']})"
                    
                    content_preview = res['content'][:150].replace('\n', ' ')
                    logger.info(f" [Source: {source}] {content_preview}...")
            else:
                logger.warning(f"No results or failure for query '{query}': {result_json}")
                
        except Exception as e:
            logger.error(f"Error during search test for '{query}': {e}")
        
        # Avoid rapid fire requests
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(test_web_search())

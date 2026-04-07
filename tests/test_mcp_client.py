import asyncio
import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.backend.infrastructure.mcp_connector.client import get_mcp_client
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_mcp_connectivity():
    """
    Test script to verify MCP Client connectivity and tool discovery.
    Run this after starting the MCP Server in SSE mode:
    python -m src.mcp_server.server --sse
    """
    if os.getenv("RUN_MCP_TESTS") != "1":
        pytest.skip("RUN_MCP_TESTS not set, skipping MCP connectivity test")
    # 1. Get client instance
    # Ensure MCP_SERVER_SSE_URL=http://localhost:8000/sse is set if testing SSE
    client = get_mcp_client()
    
    logger.info(f"Using MCP Client with SSE URL: {client.sse_url}")
    if not client.sse_url:
        logger.info(f"Using Stdio transport with path: {client.server_script_path}")

    # 2. Test tool listing
    logger.info("Listing available tools...")
    tools = await client.list_tools()
    
    if not tools:
        logger.error("No tools found! Is the server running?")
        return
    
    logger.info(f"Found {len(tools)} tools:")
    for t in tools:
        logger.info(f" - {t['name']}: {t['description'][:50]}...")

    # 3. Test a simple tool call (query_knowledge_hub)
    # Note: query_knowledge_hub is the legacy name, or use the dynamic name found
    target_tool = "query_knowledge_hub"
    if not any(t['name'] == target_tool for t in tools):
        target_tool = tools[0]['name']
        
    logger.info(f"Testing tool call: {target_tool}")
    
    # query_knowledge_hub expects 'query'
    # For others, we might need different args
    test_args = {"query": "博主为什么要搭建该项目"}
    if target_tool == "list_collections":
        test_args = {}
        
    results = await client.call_tool(target_tool, test_args)
    
    if results:
        logger.info("Successfully received results from MCP Server:")
        for res in results:
            content_preview = res['content'][:100].replace('\n', ' ')
            logger.info(f" [{res['document_name']}] {content_preview}...")
    else:
        logger.warning("Tool call returned no results (this might be normal if DB is empty)")

if __name__ == "__main__":
    asyncio.run(test_mcp_connectivity())

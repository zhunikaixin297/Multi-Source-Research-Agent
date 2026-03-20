import time
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool, StructuredTool
from pydantic import create_model, Field
from .client import get_mcp_client
from ..agents.states import RawSearchResult

logger = logging.getLogger(__name__)

# Simple rate limiting state
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

_web_search_rate_limiter = RateLimiter(calls_per_minute=5)

from ..search.factory import web_search_tool

def parse_tool_output(tool_output: str) -> List[RawSearchResult]:
    """Helper to parse the JSON string returned by the tools back into List[RawSearchResult]."""
    try:
        data = json.loads(tool_output)
        if isinstance(data, list):
            return data
        return [{"content": str(data), "document_name": "Tool Output"}]
    except Exception as e:
        logger.warning(f"Failed to parse tool output: {e}")
        return [{"content": str(tool_output), "document_name": "Tool Output"}]

async def get_dynamic_mcp_tools() -> List[StructuredTool]:
    """
    Dynamically discover and create LangChain tools from the MCP server.
    """
    mcp_client = get_mcp_client()
    try:
        mcp_tools_list = await mcp_client.list_tools()
    except Exception as e:
        logger.error(f"Failed to list MCP tools: {e}")
        return []

    lc_tools = []
    
    for tool_info in mcp_tools_list:
        tool_name = tool_info["name"]
        description = tool_info["description"]
        input_schema = tool_info["inputSchema"]
        
        # Build Pydantic model for args
        fields = {}
        properties = input_schema.get("properties", {})
        required_fields = input_schema.get("required", [])
        
        for prop_name, prop_def in properties.items():
            prop_type = str
            t = prop_def.get("type")
            if t == "integer":
                prop_type = int
            elif t == "number":
                prop_type = float
            elif t == "boolean":
                prop_type = bool
            
            # Create field with description
            default_val = ... if prop_name in required_fields else prop_def.get("default", None)
            
            # Handle optional fields explicitly
            if default_val is None:
                prop_type = Optional[prop_type]
            
            fields[prop_name] = (prop_type, Field(default=default_val, description=prop_def.get("description", "")))
            
        args_model = create_model(f"{tool_name}Input", **fields)
        
        # Define the async wrapper
        # Use a factory to capture tool_name correctly in the closure
        def make_wrapper(t_name):
            async def _wrapper(**kwargs):
                try:
                    # 过滤掉值为 None 的参数，因为 MCP Server 的 JSON Schema 可能不支持 null 类型
                    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                    
                    results = await mcp_client.call_tool(t_name, filtered_kwargs)
                    if not results:
                         return json.dumps([{"content": f"No results from {t_name}", "document_name": "System"}], ensure_ascii=False)
                    return json.dumps(results, ensure_ascii=False)
                except Exception as e:
                    return json.dumps([{"content": f"Error calling {t_name}: {e}", "document_name": "System"}], ensure_ascii=False)
            return _wrapper

        # Create StructuredTool
        tool_instance = StructuredTool.from_function(
            func=None,
            coroutine=make_wrapper(tool_name),
            name=tool_name,
            description=description,
            args_schema=args_model
        )
        lc_tools.append(tool_instance)
        
    return lc_tools

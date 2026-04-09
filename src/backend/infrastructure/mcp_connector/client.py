import asyncio
import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client
from ...core.config import settings

logger = logging.getLogger(__name__)

class KnowledgeBaseMCPClient:
    """
    MCP Client for connecting to the MODULAR-RAG-MCP-SERVER.
    Supports both stdio (local) and SSE (remote) transport.
    """
    def __init__(
        self, 
        server_script_path: Optional[str] = None, 
        python_path: Optional[str] = None,
        sse_url: Optional[str] = None
    ):
        self.sse_url = sse_url
        self.server_params = None
        
        if not sse_url:
            # Fallback to stdio
            import sys
            if not server_script_path:
                raise ValueError("Either sse_url or server_script_path must be provided")
                
            if python_path is None:
                python_path = sys.executable
                
            # The environment needs to have the server script path in PYTHONPATH
            env = os.environ.copy()
            if "PYTHONPATH" in env:
                env["PYTHONPATH"] = f"{server_script_path}:{env['PYTHONPATH']}"
            else:
                env["PYTHONPATH"] = server_script_path

            self.server_params = StdioServerParameters(
                command=python_path,
                args=["-m", "src.mcp_server.server"],
                env=env,
                cwd=server_script_path
            )
            self.server_script_path = server_script_path

    async def _get_session(self):
        """Context manager to yield a session based on transport type."""
        if self.sse_url:
            return sse_client(self.sse_url)
        else:
            return stdio_client(self.server_params)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        tools_list = []
        try:
            async with await self._get_session() as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    # Convert Tool objects to dicts
                    tools_list = [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.inputSchema
                        }
                        for tool in result.tools
                    ]
        except Exception as e:
            logger.error(f"Failed to list MCP tools: {e}")
        return tools_list

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generic method to call any MCP tool."""
        extracted_results = []
        try:
            async with await self._get_session() as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    logger.info(f"Calling MCP tool '{tool_name}' with args: {arguments}")
                    result = await session.call_tool(tool_name, arguments=arguments)
                    
                    if result.isError:
                        error_text = result.content[0].text if result.content else "Unknown error"
                        logger.error(f"MCP Tool Error: {error_text}")
                        return []
                    
                    # Parse the text contents from the MCP result
                    markdown_content = ""
                    json_references = None
                    
                    for block in result.content:
                        if block.type == "text":
                            if "**References (JSON):**" in block.text:
                                # Extract JSON block
                                try:
                                    json_str = block.text.split("```json")[1].split("```")[0].strip()
                                    json_references = json.loads(json_str)
                                except Exception as e:
                                    logger.warning(f"Failed to parse references JSON: {e}")
                            else:
                                markdown_content += block.text + "\n"
                    
                    # If we got valid results, format them for the agent
                    if markdown_content.strip():
                        raw_citations = []
                        if json_references and "citations" in json_references:
                            raw_citations = json_references["citations"]

                        # For query_knowledge_hub, expose one result per citation so
                        # downstream logs/metrics reflect real retrieved result count.
                        if tool_name == "query_knowledge_hub" and raw_citations:
                            for cit in raw_citations:
                                metadata = cit.get("metadata") or {}
                                source = cit.get("source", "Unknown MCP Source")
                                source_ref = metadata.get("source_ref")
                                doc_hash = metadata.get("doc_hash")
                                doc_tokens = []
                                if source_ref:
                                    doc_tokens.append(f"source_ref={source_ref}")
                                if doc_hash:
                                    doc_tokens.append(f"doc_hash={doc_hash}")
                                suffix = f" ({', '.join(doc_tokens)})" if doc_tokens else ""

                                extracted_results.append({
                                    "content": cit.get("text_snippet", "") or markdown_content.strip(),
                                    "document_name": f"{source}{suffix}",
                                    "url": None,
                                    "score": float(cit.get("score", 0.0)) if cit.get("score") is not None else None,
                                    "provider": "knowledge base",
                                })
                        else:
                            # Fallback for tools without citations (e.g., list_collections/get_document_summary)
                            sources = [cit.get("source", "Unknown MCP Source") for cit in raw_citations]
                            source_str = ", ".join(set(sources)) if sources else "System"
                            primary_score = None
                            if raw_citations:
                                scores = [float(cit.get("score", 0)) for cit in raw_citations if "score" in cit]
                                if scores:
                                    primary_score = max(scores)

                            extracted_results.append({
                                "content": markdown_content.strip(),
                                "document_name": source_str,
                                "url": None,
                                "score": primary_score,
                                "provider": "knowledge base",
                            })
                        
        except Exception as e:
            logger.error(f"Failed to execute MCP tool call: {e}")
            
        return extracted_results

    async def query_knowledge_hub(self, query: str, top_k: int = 5, collection: Optional[str] = None) -> List[Dict[str, str]]:
        """Legacy wrapper for query_knowledge_hub tool."""
        args = {"query": query, "top_k": top_k}
        if collection is not None:
            args["collection"] = collection
        return await self.call_tool("query_knowledge_hub", args)

# Global instance for the factory
_mcp_client: Optional[KnowledgeBaseMCPClient] = None


def _resolve_local_mcp_server_path() -> Optional[str]:
    """
    Resolve local MCP server root path.
    Priority:
    1) MCP_SERVER_LOCAL_PATH (settings)
    2) Auto-discover sibling repo: ../Enterprise-Modular-RAG-MCP
    """
    if settings.mcp.server_local_path:
        return settings.mcp.server_local_path

    repo_root = Path(__file__).resolve().parents[4]  # Multi-Source-Research-Agent
    candidate = repo_root.parent / "Enterprise-Modular-RAG-MCP"
    server_entry = candidate / "src" / "mcp_server" / "server.py"
    if server_entry.exists():
        return str(candidate)
    return None

def get_mcp_client() -> KnowledgeBaseMCPClient:
    global _mcp_client
    if _mcp_client is None:
        # Get configuration from centralized settings
        sse_url = settings.mcp.server_sse_url
        
        if sse_url:
             _mcp_client = KnowledgeBaseMCPClient(sse_url=sse_url)
        else:
            # Fallback to local stdio mode using path from settings
            server_path = _resolve_local_mcp_server_path()
            if not server_path:
                raise ValueError(
                    "MCP local server path is not configured. "
                    "Set MCP_SERVER_LOCAL_PATH or MCP_SERVER_SSE_URL."
                )
            _mcp_client = KnowledgeBaseMCPClient(server_script_path=server_path)
    return _mcp_client

from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator
import json
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event

from ..llm.factory import get_research_llm
from .states import ResearchTask, RawSearchResult
from ..mcp_connector.tools import get_dynamic_mcp_tools, parse_tool_output, web_search_tool
from .prompt.worker_prompt import WORKER_SEARCH_ROUTING_PROMPT

# State for Search Subgraph (用于包装 ReAct Agent 的输入输出)
class SearchState(TypedDict):
    task: ResearchTask
    search_results: Annotated[List[RawSearchResult], operator.add]

async def search_agent_node(state: SearchState, config: RunnableConfig) -> Dict[str, Any]:
    """
    使用 create_react_agent 构建搜索逻辑。
    它会自动处理工具调用循环 (Action -> Tool -> Observation)。
    """
    task = state["task"]
    
    # 1. 动态获取工具
    mcp_tools = await get_dynamic_mcp_tools()
    
    if not mcp_tools:
        await adispatch_custom_event(
            "worker_progress", 
            {
                "task_id": task.id,
                "title": task.title,
                "status": "researching",
                "message": "无法连接企业知识库 (MCP)，仅使用网络搜索..."
            },
            config=config
        )
        tools = [web_search_tool]
    else:
        tools = mcp_tools + [web_search_tool]
        
    llm = get_research_llm()
    
    # 2. 构建系统提示词
    system_msg = WORKER_SEARCH_ROUTING_PROMPT.format(
        query=task.query,
        intent=task.intent
    )
    
    # 3. 创建并调用 ReAct Agent
    # 在较新版本的 langgraph 中，参数名为 state_modifier
    # 如果环境版本较老，可以不使用 modifier，而是将 SystemMessage 放到初始对话中
    agent = create_react_agent(llm, tools=tools)
    
    # 初始化对话，让模型开始搜索，同时传递系统提示词
    initial_messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"请根据系统提示词，开始执行搜索任务。查询词: {task.query}")
    ]
    
    response = await agent.ainvoke({"messages": initial_messages}, config=config)
    messages = response.get("messages", [])
    
    # 4. 从消息历史中提取最后一轮工具调用的结果
    # 逻辑：从后往前找，提取最后一段连续的 ToolMessage
    # 这样可以确保 Agent 在多次尝试后，返回的是它认为最相关的最新结果
    new_results = []
    found_tool_msg = False
    
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            found_tool_msg = True
            # 将工具返回的字符串解析为标准格式
            parsed = parse_tool_output(msg.content)
            # 由于是倒序遍历，我们需要将结果插入到列表头部以保持顺序
            new_results = parsed + new_results
        elif found_tool_msg:
            # 一旦遇到非 ToolMessage 且之前已经找到了 ToolMessage，说明这一轮工具调用提取结束
            break
            
    return {"search_results": new_results}

# Graph Construction
def get_search_subgraph():
    """
    将 ReAct Agent 包装为一个普通的 Node，放入 StateGraph 中。
    这样可以保持与外部系统的 State (SearchState) 兼容。
    """
    workflow = StateGraph(SearchState)
    
    workflow.add_node("search_agent", search_agent_node)
    workflow.add_edge(START, "search_agent")
    workflow.add_edge("search_agent", END)
    
    return workflow.compile()

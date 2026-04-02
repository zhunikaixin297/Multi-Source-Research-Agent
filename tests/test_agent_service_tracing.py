import os
import sys
from typing import Any, Dict, List, TypedDict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from src.backend.domain.models import ReportRequest
from src.backend.services.agent_service import _build_langfuse_trace_config


class NoOpLangfuseHandler(BaseCallbackHandler):
    pass


class FakeLangfuseClient:
    def auth_check(self) -> bool:
        return True

    def flush(self) -> None:
        return None

    def get_prompt(self, *args, **kwargs):
        raise RuntimeError("prompt fetch disabled in test")


class FakeResearchLLM:
    def __init__(self):
        self.invocations: List[Dict[str, Any]] = []

    async def ainvoke(self, messages, config=None):
        metadata = dict(config.get("metadata", {})) if config else {}
        configurable = dict(config.get("configurable", {})) if config else {}
        tags = list(config.get("tags", [])) if config else []
        self.invocations.append(
            {
                "messages": messages,
                "config": {
                    "metadata": metadata,
                    "configurable": configurable,
                    "run_name": config.get("run_name") if config else None,
                    "tags": tags,
                },
            }
        )

        call_index = len(self.invocations)
        if call_index == 1:
            return AIMessage(
                content='[{"title": "Topic mapping", "intent": "research", "query": "collect source material"}]'
            )
        if call_index == 2:
            return AIMessage(content='{"content": "Worker content", "summary": "Worker summary"}')
        if call_index == 3:
            return AIMessage(content='{"is_sufficient": true, "knowledge_gap": ""}')
        return AIMessage(content="Final report content")


class FakeSearchState(TypedDict):
    goal: str
    task: Any
    search_results: List[Dict[str, Any]]


def _build_fake_search_graph(search_invocations: List[Dict[str, Any]]):
    async def fake_search_node(state: FakeSearchState, config):
        search_invocations.append(
            {
                "metadata": dict(config.get("metadata", {})) if config else {},
                "configurable": dict(config.get("configurable", {})) if config else {},
                "run_name": config.get("run_name") if config else None,
                "tags": list(config.get("tags", [])) if config else [],
            }
        )
        return {
            "search_results": [
                {
                    "content": "workspace result",
                    "document_name": "Doc 1",
                    "provider": "session_rag",
                }
            ]
        }

    workflow = StateGraph(FakeSearchState)
    workflow.add_node("search_agent", fake_search_node)
    workflow.add_edge(START, "search_agent")
    workflow.add_edge("search_agent", END)
    return workflow.compile()


def test_build_langfuse_trace_config_uses_thread_id_as_session_anchor():
    config = _build_langfuse_trace_config(
        thread_id="thread-123",
        workspace_id="workspace-456",
        action="start",
        trace_name="research-report-thread-123",
        extra_tags=["goal:demo", "workspace:workspace-456", None],
    )

    assert config["run_name"] == "research-report-thread-123"
    assert config["tags"] == [
        "deepresearch-agent",
        "workspace:workspace-456",
        "thread:thread-123",
        "action:start",
        "goal:demo",
    ]

    metadata = config["metadata"]
    assert metadata["langfuse_session_id"] == "thread-123"
    assert metadata["workspace_id"] == "workspace-456"
    assert metadata["thread_id"] == "thread-123"
    assert metadata["action"] == "start"
    assert metadata["trace_name"] == "research-report-thread-123"
    assert metadata["langfuse_tags"] == config["tags"]
    assert "langfuse_user_id" not in metadata


def test_build_langfuse_trace_config_can_include_explicit_user_id():
    config = _build_langfuse_trace_config(
        thread_id="thread-789",
        workspace_id="workspace-456",
        action="revise",
        trace_name="research-report-thread-789",
        user_id="user-001",
    )

    assert config["metadata"]["langfuse_user_id"] == "user-001"
    assert config["metadata"]["langfuse_session_id"] == "thread-789"


@pytest.mark.asyncio
async def test_generate_report_runs_full_graph_and_propagates_trace_context(monkeypatch):
    from src.backend.infrastructure.agents import orchestrator_agent
    from src.backend.infrastructure.agents import planner_agent
    from src.backend.infrastructure.agents import reflector_agent
    from src.backend.infrastructure.agents import utils as agent_utils
    from src.backend.infrastructure.agents import worker_agent
    from src.backend.services import agent_service as agent_service_module

    fake_llm = FakeResearchLLM()
    search_invocations: List[Dict[str, Any]] = []
    fake_search_graph = _build_fake_search_graph(search_invocations)

    fake_client = FakeLangfuseClient()

    monkeypatch.setattr(agent_service_module, "init_langfuse_client", lambda **kwargs: fake_client)
    monkeypatch.setattr(agent_utils, "init_langfuse_client", lambda **kwargs: fake_client)
    monkeypatch.setattr(agent_service_module, "CallbackHandler", lambda: NoOpLangfuseHandler())

    monkeypatch.setattr(planner_agent, "get_research_llm", lambda: fake_llm)
    monkeypatch.setattr(worker_agent, "get_research_llm", lambda: fake_llm)
    monkeypatch.setattr(reflector_agent, "get_research_llm", lambda: fake_llm)
    monkeypatch.setattr(orchestrator_agent, "get_research_llm", lambda: fake_llm)
    monkeypatch.setattr(worker_agent, "get_search_subgraph", lambda: fake_search_graph)

    service = agent_service_module.AgentServiceImpl()
    thread_id = "thread-full-001"
    workspace_id = "workspace-full-001"

    start_request = ReportRequest(
        task_id=thread_id,
        workspace_id=workspace_id,
        query="Analyze the architecture",
        action="start",
    )
    start_events = [event async for event in service.generate_report(start_request)]
    assert any("event: interrupt" in event for event in start_events)

    resume_request = ReportRequest(
        task_id=thread_id,
        workspace_id=workspace_id,
        action="approve",
    )
    resume_events = [event async for event in service.generate_report(resume_request)]
    assert any("event: done" in event for event in resume_events)

    assert len(fake_llm.invocations) == 4
    for invocation in fake_llm.invocations:
        config = invocation["config"]
        assert config["configurable"]["thread_id"] == thread_id
        assert config["configurable"]["workspace_id"] == workspace_id
        assert config["metadata"]["langfuse_session_id"] == thread_id
        assert config["metadata"]["thread_id"] == thread_id
        assert config["metadata"]["workspace_id"] == workspace_id
        assert config["metadata"]["trace_name"] == f"research-report-{thread_id}"
        assert f"workspace:{workspace_id}" in config["metadata"]["langfuse_tags"]
        assert f"thread:{thread_id}" in config["metadata"]["langfuse_tags"]

    assert len(search_invocations) == 1
    search_config = search_invocations[0]
    assert search_config["configurable"]["thread_id"] == thread_id
    assert search_config["configurable"]["workspace_id"] == workspace_id
    assert search_config["metadata"]["langfuse_session_id"] == thread_id
    assert search_config["metadata"]["trace_name"] == f"research-report-{thread_id}"

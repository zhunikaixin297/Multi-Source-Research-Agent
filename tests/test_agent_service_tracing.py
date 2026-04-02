import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.backend.services.agent_service import _build_langfuse_trace_config


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

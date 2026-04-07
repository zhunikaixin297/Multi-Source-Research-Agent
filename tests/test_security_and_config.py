import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.backend.api import server as api_server
from src.backend.infrastructure.parse.parser import DoclingParser
from src.backend.infrastructure.repository.opensearch_store import AsyncOpenSearchRAGStore
from src.backend.domain.models import RetrievedChunk, DocumentChunk
from src.backend.infrastructure.llm import factory as llm_factory


def test_sanitize_upload_filename_strips_path():
    assert api_server.sanitize_upload_filename("../evil.pdf") == "evil.pdf"
    assert api_server.sanitize_upload_filename("..\\evil.pdf") == "evil.pdf"


def test_ensure_allowed_extension_blocks_unknown():
    with pytest.raises(Exception):
        api_server.ensure_allowed_extension("malware.exe")


def test_resolve_cors_settings_disables_credentials_on_wildcard():
    origins, allow_credentials = api_server.resolve_cors_settings(["*"], True)
    assert origins == ["*"]
    assert allow_credentials is False


def test_docling_images_export_uses_images_dir(tmp_path):
    class DummySessionManager:
        def init_workspace_env(self, workspace_id):
            return tmp_path / "root"

        def workspace_images_dir(self, workspace_id):
            images_dir = tmp_path / "root" / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            return images_dir

    class DummyPicture:
        def get_image(self, doc):
            class DummyImage:
                def __init__(self):
                    self.saved_path = None

                def save(self, path, format=None):
                    self.saved_path = path
            return DummyImage()

    dummy_doc = SimpleNamespace(pictures=[DummyPicture()])
    parser = DoclingParser(converter=SimpleNamespace(), session_manager=DummySessionManager())

    images = parser._export_workspace_images(dummy_doc, "ws1", "doc123")
    assert images
    assert "/images/" in images[0]["path"].replace("\\", "/")


@pytest.mark.asyncio
async def test_opensearch_hybrid_search_skips_vectors_when_embedding_missing():
    store = AsyncOpenSearchRAGStore.__new__(AsyncOpenSearchRAGStore)

    async def _get_embedding_async(text):
        return None

    async def bm25_search(query_text, k=5):
        return [{"_id": "c1", "_source": {"chunk_id": "c1", "document_id": "d1", "content": "x"}}]

    async def _base_vector_search(*args, **kwargs):
        raise AssertionError("Vector search should be skipped when embedding is None")

    class DummyClient:
        async def mget(self, index, body):
            return {"docs": [{"_id": "c1", "found": True, "_source": {"chunk_id": "c1", "document_id": "d1", "content": "x"}}]}

    store._get_embedding_async = _get_embedding_async
    store.bm25_search = bm25_search
    store._base_vector_search = _base_vector_search
    store.client = DummyClient()
    store.index_name = "idx"

    results = await AsyncOpenSearchRAGStore.hybrid_search(store, "q", k=1, rrf_k=60)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_report_node_returns_empty_report_on_error(monkeypatch):
    async def failing_invoke(*args, **kwargs):
        raise RuntimeError("boom")

    class DummyLLM:
        async def ainvoke(self, *args, **kwargs):
            return await failing_invoke()

    monkeypatch.setattr("src.backend.infrastructure.agents.orchestrator_agent.get_research_llm", lambda: DummyLLM())
    async def _noop_event(*args, **kwargs):
        return None
    monkeypatch.setattr("src.backend.infrastructure.agents.orchestrator_agent.adispatch_custom_event", _noop_event)

    from src.backend.infrastructure.agents.orchestrator_agent import report_node

    state = {"goal": "g", "results": []}
    result = await report_node(state, config=None)
    assert result == {"final_report": ""}


def test_llm_factory_raises_when_missing_config(monkeypatch):
    class DummyConfig:
        base_url = None
        api_key = None
        model = None
        max_concurrency = 1

    class DummySettings:
        def get_llm_config_by_name(self, name):
            return DummyConfig()

    monkeypatch.setattr(llm_factory, "settings", DummySettings(), raising=True)

    with pytest.raises(RuntimeError):
        llm_factory._create_chat_llm("research")

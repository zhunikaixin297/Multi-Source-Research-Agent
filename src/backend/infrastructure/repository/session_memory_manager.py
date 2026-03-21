import logging
import shutil
import threading
import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from ...core.config import BASE_DIR, SessionRAGSettings
from ...domain.models import DocumentChunk

log = logging.getLogger(__name__)


class SessionMemoryManager:
    _client_lock = threading.Lock()
    _client: Optional[chromadb.PersistentClient] = None

    def __init__(self, settings: SessionRAGSettings) -> None:
        self._settings = settings
        self._chroma_path = BASE_DIR / settings.chroma_path
        self._workspaces_root = BASE_DIR / settings.workspaces_root
        self._chroma_path.mkdir(parents=True, exist_ok=True)
        self._workspaces_root.mkdir(parents=True, exist_ok=True)

    def _get_client(self) -> chromadb.PersistentClient:
        if self.__class__._client is None:
            with self.__class__._client_lock:
                if self.__class__._client is None:
                    self.__class__._client = chromadb.PersistentClient(
                        path=str(self._chroma_path),
                        settings=ChromaSettings(
                            anonymized_telemetry=False,
                            allow_reset=False,
                        ),
                    )
        return self.__class__._client

    @staticmethod
    def collection_name(workspace_id: str) -> str:
        return f"ws_{workspace_id}"

    def get_or_create_collection(self, workspace_id: str) -> Any:
        client = self._get_client()
        return client.get_or_create_collection(
            name=self.collection_name(workspace_id),
            metadata={"hnsw:space": self._settings.distance_metric},
        )

    def get_or_create_workspace(self, workspace_id: str) -> Any:
        self.init_workspace_env(workspace_id)
        self.workspace_documents_dir(workspace_id).mkdir(parents=True, exist_ok=True)
        return self.get_or_create_collection(workspace_id)

    def get_collection_if_exists(self, workspace_id: str) -> Optional[Any]:
        client = self._get_client()
        try:
            return client.get_collection(name=self.collection_name(workspace_id))
        except Exception:
            return None

    def init_workspace_env(self, workspace_id: str) -> Path:
        images_dir = self.workspace_images_dir(workspace_id)
        docs_dir = self.workspace_documents_dir(workspace_id)
        images_dir.mkdir(parents=True, exist_ok=True)
        docs_dir.mkdir(parents=True, exist_ok=True)
        return self.workspace_root_dir(workspace_id)

    def workspace_root_dir(self, workspace_id: str) -> Path:
        return self._workspaces_root / workspace_id

    def workspace_images_dir(self, workspace_id: str) -> Path:
        return self.workspace_root_dir(workspace_id) / "images"

    def workspace_documents_dir(self, workspace_id: str) -> Path:
        return self.workspace_root_dir(workspace_id) / "documents"

    def list_workspace_documents(self, workspace_id: str) -> list[Path]:
        docs_dir = self.workspace_documents_dir(workspace_id)
        if not docs_dir.exists():
            return []
        return [p for p in docs_dir.iterdir() if p.is_file()]

    def workspace_document_count(self, workspace_id: str) -> int:
        return len(self.list_workspace_documents(workspace_id))

    def cleanup_workspace(self, workspace_id: str) -> None:
        client = self._get_client()
        collection_name = self.collection_name(workspace_id)
        workspace_dir = self.workspace_root_dir(workspace_id)
        try:
            client.delete_collection(name=collection_name)
        except Exception as e:
            log.warning(f"删除工作区集合失败 workspace_id={workspace_id}: {e}")
        try:
            if workspace_dir.exists():
                shutil.rmtree(workspace_dir)
        except Exception as e:
            log.warning(f"删除工作区目录失败 workspace_id={workspace_id}: {e}")

    def max_workspace_documents(self) -> int:
        return self._settings.max_workspace_documents

    async def add_documents(self, workspace_id: str, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        collection = self.get_or_create_workspace(workspace_id)
        from ..llm.factory import get_embedding_model
        embedding_model = get_embedding_model()
        texts = [chunk.content for chunk in chunks]
        if hasattr(embedding_model, "aembed_documents"):
            embeddings = await embedding_model.aembed_documents(texts)
        else:
            embeddings = await asyncio.to_thread(embedding_model.embed_documents, texts)

        def _serialize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
            serialized: dict[str, Any] = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    serialized[key] = value
                elif value is None:
                    continue
                else:
                    serialized[key] = json.dumps(value, ensure_ascii=False)
            return serialized

        metadatas = [
            _serialize_metadata(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "parent_headings": chunk.parent_headings,
                    "metadata": chunk.metadata,
                }
            )
            for chunk in chunks
        ]
        ids = [chunk.chunk_id for chunk in chunks]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)


_session_memory_manager: Optional[SessionMemoryManager] = None
_session_memory_manager_lock = threading.Lock()


def get_session_memory_manager(settings: Optional[SessionRAGSettings] = None) -> SessionMemoryManager:
    global _session_memory_manager
    if _session_memory_manager is None:
        with _session_memory_manager_lock:
            if _session_memory_manager is None:
                if settings is None:
                    from ...core.config import settings as global_settings
                    settings = global_settings.session_rag
                _session_memory_manager = SessionMemoryManager(settings)
    return _session_memory_manager

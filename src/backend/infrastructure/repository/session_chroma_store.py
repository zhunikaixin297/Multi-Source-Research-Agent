import json
import logging
import asyncio
from typing import Any, Dict, List, Optional

from ...core.config import SessionRAGSettings
from ...domain.interfaces import EmbeddingModel
from ...domain.models import DocumentChunk, RetrievedChunk
from .session_memory_manager import SessionMemoryManager

log = logging.getLogger(__name__)


class SessionChromaStore:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        session_manager: SessionMemoryManager,
        settings: SessionRAGSettings,
    ) -> None:
        self.embedding_model = embedding_model
        self.session_manager = session_manager
        self.settings = settings

    @staticmethod
    def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        serialized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                serialized[key] = value
            elif value is None:
                continue
            else:
                serialized[key] = json.dumps(value, ensure_ascii=False)
        return serialized

    @staticmethod
    def _deserialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        restored: Dict[str, Any] = {}
        for key, value in metadata.items():
            if not isinstance(value, str):
                restored[key] = value
                continue
            text = value.strip()
            if (text.startswith("{") and text.endswith("}")) or (
                text.startswith("[") and text.endswith("]")
            ):
                try:
                    restored[key] = json.loads(value)
                    continue
                except Exception:
                    pass
            restored[key] = value
        return restored

    async def add_chunks(self, workspace_id: str, chunks: List[DocumentChunk]) -> int:
        return await self.session_manager.add_documents(workspace_id, chunks)

    async def search(
        self, workspace_id: str, query: str, top_k: Optional[int] = None
    ) -> List[RetrievedChunk]:
        top_k = top_k or self.settings.top_k
        collection = self.session_manager.get_collection_if_exists(workspace_id)
        if collection is None:
            return []
        if collection.count() == 0:
            return []
        if hasattr(self.embedding_model, "aembed_query"):
            query_embedding = await self.embedding_model.aembed_query(query)
        else:
            query_embedding = await asyncio.to_thread(self.embedding_model.embed_query, query)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        output: List[RetrievedChunk] = []
        for idx, chunk_id in enumerate(ids):
            metadata_raw = metadatas[idx] or {}
            metadata_full = self._deserialize_metadata(metadata_raw)
            chunk_metadata = metadata_full.get("metadata", {})
            chunk = DocumentChunk(
                chunk_id=metadata_full.get("chunk_id", chunk_id),
                document_id=metadata_full.get("document_id", ""),
                document_name=metadata_full.get("document_name", ""),
                content=documents[idx] or "",
                parent_headings=metadata_full.get("parent_headings", []),
                metadata=chunk_metadata if isinstance(chunk_metadata, dict) else {},
            )
            distance = distances[idx] if idx < len(distances) else 0.0
            score = max(0.0, 1.0 - float(distance))
            output.append(
                RetrievedChunk(
                    chunk=chunk,
                    search_score=score,
                )
            )
        return output

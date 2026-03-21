import logging
import asyncio
from typing import Awaitable, Callable, List, Optional

from ..domain.interfaces import DocumentParser, TextSplitter
from ..domain.models import DocumentSource, RetrievedChunk
from ..infrastructure.llm.reranker import TEIRerankerClient
from ..infrastructure.repository.session_chroma_store import SessionChromaStore
from ..infrastructure.repository.session_memory_manager import SessionMemoryManager

log = logging.getLogger(__name__)


class SessionIngestionService:
    def __init__(
        self,
        parser: DocumentParser,
        splitter: TextSplitter,
        store: SessionChromaStore,
        session_manager: SessionMemoryManager,
        max_concurrency: int,
    ) -> None:
        self.parser = parser
        self.splitter = splitter
        self.store = store
        self.session_manager = session_manager
        self.docling_semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def pipeline(
        self,
        source: DocumentSource,
        workspace_id: str,
        status_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> int:
        async def emit(message: str) -> None:
            if status_callback:
                await status_callback(message)

        await emit(f"--- [工作区摄取开始] 文档: {source.document_name} ---")
        async with self.docling_semaphore:
            self.session_manager.init_workspace_env(workspace_id)
            source.metadata = source.metadata or {}
            source.metadata["workspace_id"] = workspace_id
            markdown = await self.parser.parse(source)
            if not markdown or not markdown.strip():
                await emit("解析失败: 未提取到内容。")
                return 0
            chunks = self.splitter.split(markdown, source)
            if not chunks:
                await emit("切分失败或未产生任何块。")
                return 0
            written = await self.store.add_chunks(workspace_id, chunks)
            await emit(f"工作区向量写入完成，共写入 {written} 个文本块。")
            return written


class SessionRetrievalService:
    def __init__(self, store: SessionChromaStore, reranker: TEIRerankerClient) -> None:
        self.store = store
        self.reranker = reranker

    async def retrieve(self, workspace_id: str, query: str) -> List[RetrievedChunk]:
        recalled = await self.store.search(
            workspace_id=workspace_id, query=query, top_k=self.store.settings.top_k
        )
        if not recalled:
            return []
        reranked = await self.reranker.arerank(
            query=query, chunks=recalled, top_n=self.store.settings.top_n
        )
        return reranked if reranked else recalled[: self.store.settings.top_n]

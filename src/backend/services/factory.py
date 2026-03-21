import logging
import asyncio
from functools import lru_cache

from ..domain.interfaces import AgentService
# 1. 导入 Service 类
from .ingestion_service import IngestionService
from .agent_service import AgentServiceImpl
from .session_rag_service import SessionIngestionService, SessionRetrievalService

# 2. 导入其他基础设施的工厂函数
from ..infrastructure.parse.factory import (
    get_docling_parser,
    get_llm_preprocessor,
    get_markdown_splitter
)
from ..infrastructure.repository.factory import get_opensearch_store, get_session_store
from ..infrastructure.repository.session_memory_manager import get_session_memory_manager
from ..infrastructure.llm.factory import get_rerank_client
from ..core.config import settings

log = logging.getLogger(__name__)


@lru_cache()
def get_ingestion_service() -> IngestionService:
    """
    [工厂方法] 组装并获取 IngestionService 单例。
    
    职责：
    1. 调用底层组件的工厂方法获取实例。
    2. 将这些实例注入到 IngestionService 中。
    3. 返回组装好的 Service。
    """
    log.info("正在组装 IngestionService...")
    
    try:
        # 获取依赖实例
        parser_instance = get_docling_parser()
        splitter_instance = get_markdown_splitter()
        preprocessor_instance = get_llm_preprocessor()
        store_instance = get_opensearch_store()
        
        # 注入依赖并实例化
        service = IngestionService(
            parser=parser_instance,
            splitter=splitter_instance,
            preprocessor=preprocessor_instance,
            store=store_instance,
            max_concurrency=settings.docling_general.max_concurrent_docs
        )
        
        return service
        
    except Exception as e:
        log.error(f"IngestionService 工厂初始化失败: {e}", exc_info=True)
        raise e


@lru_cache()
def get_agent_service() -> AgentService:
    """
    创建并返回 AgentService 的实例 (返回接口类型，实际是实现类)
    """
    return AgentServiceImpl()


@lru_cache()
def get_session_ingestion_service() -> SessionIngestionService:
    return SessionIngestionService(
        parser=get_docling_parser(),
        splitter=get_markdown_splitter(),
        store=get_session_store(),
        session_manager=get_session_memory_manager(settings.session_rag),
        max_concurrency=settings.session_rag.ingestion_max_concurrency,
    )


@lru_cache()
def get_workspace_ingestion_service() -> IngestionService:
    return IngestionService(
        parser=get_docling_parser(),
        splitter=get_markdown_splitter(),
        preprocessor=None,
        store=None,
        session_manager=get_session_memory_manager(settings.session_rag),
        max_concurrency=settings.session_rag.ingestion_max_concurrency,
    )


@lru_cache()
def get_session_retrieval_service() -> SessionRetrievalService:
    return SessionRetrievalService(
        store=get_session_store(),
        reranker=get_rerank_client(),
    )

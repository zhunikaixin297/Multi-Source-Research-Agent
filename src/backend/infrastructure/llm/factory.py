import asyncio
from functools import lru_cache
from typing import Optional, Dict, Any, AsyncIterator

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGenerationChunk

# 导入配置
from ...core.config import settings
# 导入自定义的 Reranker Client 类
from .reranker import TEIRerankerClient 

# 全局信号量缓存，用于控制不同用途 LLM 的并发
_semaphores: Dict[str, asyncio.Semaphore] = {}

def get_semaphore_by_name(name: str) -> asyncio.Semaphore:
    """获取或创建指定名称的信号量"""
    if name not in _semaphores:
        config = settings.get_llm_config_by_name(name)
        _semaphores[name] = asyncio.Semaphore(config.max_concurrency)
    return _semaphores[name]

class ConcurrencyControlledChatOpenAI(ChatOpenAI):
    """
    包装 ChatOpenAI 以支持全局并发控制。
    """
    config_name: str = ""

    async def _agenerate(self, *args, **kwargs) -> ChatResult:
        sem = get_semaphore_by_name(self.config_name)
        async with sem:
            return await super()._agenerate(*args, **kwargs)

    async def _astream(self, *args, **kwargs) -> AsyncIterator[ChatGenerationChunk]:
        sem = get_semaphore_by_name(self.config_name)
        async with sem:
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk

# ==========================================
#  通用构建辅助函数 (核心解耦逻辑)
# ==========================================
def _create_chat_llm(config_name: str, temperature: float = 0, max_retries: int = 3) -> Any:
    """
    私有辅助函数：根据配置名称动态创建支持并发控制的 LLM 实例。
    """
    # 1. 动态获取配置对象
    config = settings.get_llm_config_by_name(config_name)
    
    # 2. 实例化包装类
    return ConcurrencyControlledChatOpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        temperature=temperature,
        max_retries=max_retries,
        config_name=config_name
    )

# ==========================================
# 1. 预处理 LLM (Preprocessing LLM)
# ==========================================
@lru_cache()
def get_preprocessing_llm() -> ChatOpenAI:
    """
    获取用于文本分块预处理的 LLM 客户端单例。
    """
    return _create_chat_llm("preprocess", temperature=0)

# ==========================================
# 2. Embedding 模型 (Embedding Model)
# ==========================================
@lru_cache()
def get_embedding_model() -> OpenAIEmbeddings:
    """
    获取 Embedding 模型客户端单例。
    """
    # 1. 获取 Embedding 专用配置
    config = settings.get_llm_config_by_name("embedding")
    
    # 2. 实例化 OpenAIEmbeddings
    # 注意：OpenAIEmbeddings 的参数与 ChatOpenAI 略有不同
    return OpenAIEmbeddings(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key
    )

# ==========================================
# 3. 查询重写 LLM (Rewrite LLM)
# ==========================================
@lru_cache()
def get_rewrite_llm() -> ChatOpenAI:
    """
    获取用于 Query Rewrite 的 LLM 客户端单例。
    """
    return _create_chat_llm("rewrite", temperature=0)

# ==========================================
# 4. research LLM
# ==========================================
@lru_cache()
def get_research_llm() -> ChatOpenAI:
    """
    获取用于 research 的 LLM 客户端单例。
    """
    return _create_chat_llm("research", temperature=0)

# ==========================================
# 5. Reranker 客户端 (TEI Reranker)
# ==========================================
@lru_cache()
def get_rerank_client() -> TEIRerankerClient:
    """
    获取 TEI Reranker 客户端单例。
    注：Reranker 配置结构特殊（包含 timeout 等），且通常不视为标准 LLM，
    因此这里直接访问 settings.tei_rerank 。
    """
    return TEIRerankerClient(
        base_url=settings.tei_rerank.base_url,
        api_key=settings.tei_rerank.api_key,
        timeout=settings.tei_rerank.timeout,
        max_concurrency=settings.tei_rerank.max_concurrency
    )

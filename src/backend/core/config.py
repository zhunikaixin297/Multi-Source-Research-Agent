import os
from pathlib import Path
from typing import Literal, List, Tuple, Optional, Dict, Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- 路径配置 ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"


class BaseConfigSettings(BaseSettings):
    """
    基础配置类，定义通用的加载行为。
    """
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding='utf-8',
        extra="ignore",           # 忽略多余字段
        frozen=True,              # 不可变
        case_sensitive=False,     # 大小写不敏感
    )


# =============================================================================
#  1. 统一 LLM Provider 配置抽象
# =============================================================================

class LLMProviderConfig(BaseConfigSettings):
    """
    统一的 LLM 提供商配置基类。
    所有涉及 LLM 调用的配置都应继承此类，以保证接口一致性。
    """
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    max_concurrency: int = 3  # 默认并发数

    # 允许子类定义额外字段（如 embedding 的 dimension）
    # 但在基类层面，主要关注以上四个核心字段


# =============================================================================
#  2. 具体用途的 LLM 配置 (继承自 LLMProviderConfig)
# =============================================================================

class DoclingVLMSettings(LLMProviderConfig):
    """Docling 视觉模型配置 (DOCLING_VLM_*)"""
    model_config = SettingsConfigDict(env_prefix="DOCLING_VLM_")
    max_concurrency: int = 3


class DoclingLLMSettings(LLMProviderConfig):
    """Docling 文本模型配置 (DOCLING_LLM_*)"""
    model_config = SettingsConfigDict(env_prefix="DOCLING_LLM_")
    max_concurrency: int = 3


class PreprocessingLLMSettings(LLMProviderConfig):
    """预处理模型配置 (PREPROCESSING_LLM_*)"""
    model_config = SettingsConfigDict(env_prefix="PREPROCESSING_LLM_")
    max_concurrency: int = 3


class RewriteLLMSettings(LLMProviderConfig):
    """查询重写模型配置 (REWRITE_LLM_*)"""
    model_config = SettingsConfigDict(env_prefix="REWRITE_LLM_")
    max_concurrency: int = 30  # 高并发需求


class ResearchLLMSettings(LLMProviderConfig):
    """深度研究模型配置 (RESEARCH_LLM_*)"""
    model_config = SettingsConfigDict(env_prefix="RESEARCH_LLM_")
    max_concurrency: int = 30  # 高并发需求


class EmbeddingLLMSettings(LLMProviderConfig):
    """
    向量化模型配置 (EMBEDDING_LLM_*)
    * 特殊: 增加了 dimension 字段
    """
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_LLM_")
    dimension: int = 2560
    max_concurrency: int = 5


# =============================================================================
#  3. 其他非 LLM 类配置
# =============================================================================

class DoclingGeneralSettings(BaseConfigSettings):
    """Docling 通用行为配置 (DOCLING_*)"""
    model_config = SettingsConfigDict(env_prefix="DOCLING_")

    images_scale: float = 2.0
    do_formula_recognition: bool = True
    do_table_enrichment: bool = True
    do_pic_enrichment: bool = True
    do_ocr: bool = False
    
    accelerator_device: str = "CPU"
    accelerator_num_threads: int = 4
    max_concurrent_docs: int = 1


class ServerSettings(BaseConfigSettings):
    """API 服务器配置 (SERVER_*)"""
    model_config = SettingsConfigDict(env_prefix="SERVER_")
    
    host: str = "0.0.0.0"
    port: int = 8002
    max_file_size_mb: int = 100
    cors_allow_origins: str = "*"
    cors_allow_credentials: bool = False


class TextSplitterSettings(BaseConfigSettings):
    """文本切分配置 (使用 alias 映射旧环境变量)"""
    max_chunk_tokens: int = Field(default=1024, validation_alias="MAX_CHUNK_TOKENS")
    encoding_name: str = Field(default="cl100k_base", validation_alias="ENCODING_NAME")
    chunk_overlap_tokens: int = Field(default=100, validation_alias="CHUNK_OVERLAP_TOKENS")
    
    headers_to_split_on: List[Tuple[str, str]] = Field(
        default=[
            ("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"),
            ("####", "Header 4"), ("#####", "Header 5"), ("######", "Header 6"),
            ("#######", "Header 7"), ("########", "Header 8")
        ],
        validation_alias="HEADERS_TO_SPLIT_ON"
    )


class TeiRerankSettings(BaseConfigSettings):
    """TEI Reranker 配置"""
    model_config = SettingsConfigDict(env_prefix="TEI_RERANK_")
    
    base_url: str = "http://localhost:8082"
    api_key: Optional[str] = None
    max_concurrency: int = 50
    timeout: float = 30.0


class OpenSearchSettings(BaseConfigSettings):
    """OpenSearch 配置"""
    model_config = SettingsConfigDict(env_prefix="OPENSEARCH_")
    
    index_name: str = "rag_system_chunks_async"
    host: str = 'localhost'
    port: int = 9200
    auth: str = Field(default='admin:admin', validation_alias="AUTH")
    use_ssl: bool = False
    verify_certs: bool = False
    bulk_chunk_size: int = 500


class LangfuseSettings(BaseConfigSettings):
    """Langfuse 监控配置"""
    model_config = SettingsConfigDict(env_prefix="LANGFUSE_")
    
    secret_key: Optional[str] = None
    public_key: Optional[str] = None
    base_url: str = "http://localhost:3000"


class MCPSettings(BaseConfigSettings):
    """MCP 连接配置"""
    model_config = SettingsConfigDict(env_prefix="MCP_")
    
    # SSE 模式连接地址 (例如 http://localhost:8000/sse)
    # 如果为空，则回退到本地 Stdio 启动模式
    server_sse_url: Optional[str] = None
    
    # Stdio 模式下的本地路径 (Enterprise-Modular-RAG-MCP 项目根目录)。
    # 建议通过环境变量 MCP_SERVER_LOCAL_PATH 显式配置。
    server_local_path: Optional[str] = None


class SearchSettings(BaseConfigSettings):
    """
    Web Search 配置 (SEARCH_*)
    支持 'duckduckgo' 或 'tavily'
    """
    model_config = SettingsConfigDict(env_prefix="SEARCH_")
    
    provider: Literal["duckduckgo", "tavily"] = "duckduckgo"
    api_key: Optional[str] = None  # Tavily 需要 API Key
    max_results: int = 5
    # DuckDuckGo 后端引擎: auto, brave, duckduckgo, google, grokipedia, mojeek, wikipedia, yahoo, yandex
    ddg_backend: str = "auto"


class SessionRAGSettings(BaseConfigSettings):
    """会话级 RAG 配置 (SESSION_RAG_*)"""
    model_config = SettingsConfigDict(env_prefix="SESSION_RAG_")
    
    chroma_path: str = "data/chroma_workspaces"
    workspaces_root: str = "data/workspaces"
    top_k: int = 10
    top_n: int = 5
    max_workspace_documents: int = 9
    ingestion_max_concurrency: int = 2
    distance_metric: str = "cosine"
    workspace_ttl: int = 86400  # 默认 24 小时
    workspace_registry_db: str = "data/workspaces_registry.db"


# =============================================================================
#  4. 主配置聚合类
# =============================================================================

class Settings(BaseConfigSettings):
    """
    主配置类，聚合所有子配置。
    """
    # --- 全局 ---
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    litellm_proxy_url: Optional[str] = Field(default=None, validation_alias="LITELLM_PROXY_URL")
    hf_endpoint: str = Field(default="https://hf-mirror.com", validation_alias="HF_ENDPOINT")

    # --- 模块 ---
    server: ServerSettings = Field(default_factory=ServerSettings)
    docling_vlm: DoclingVLMSettings = Field(default_factory=DoclingVLMSettings)
    docling_llm: DoclingLLMSettings = Field(default_factory=DoclingLLMSettings)
    docling_general: DoclingGeneralSettings = Field(default_factory=DoclingGeneralSettings)
    
    splitter: TextSplitterSettings = Field(default_factory=TextSplitterSettings)
    
    # LLM 实例
    preprocessing_llm: PreprocessingLLMSettings = Field(default_factory=PreprocessingLLMSettings)
    embedding_llm: EmbeddingLLMSettings = Field(default_factory=EmbeddingLLMSettings)
    rewrite_llm: RewriteLLMSettings = Field(default_factory=RewriteLLMSettings)
    research_llm : ResearchLLMSettings = Field(default_factory=ResearchLLMSettings)
    
    tei_rerank: TeiRerankSettings = Field(default_factory=TeiRerankSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    session_rag: SessionRAGSettings = Field(default_factory=SessionRAGSettings)

    def get_llm_config_by_name(self, name: str) -> LLMProviderConfig:
        """
        [工厂方法支持] 
        根据名称动态获取 LLM 配置实例，用于解耦 LLM Factory 的调用。
        
        Args:
            name: 配置名称，例如 'rewrite', 'embedding', 'research', 'preprocess'
        
        Returns:
            LLMProviderConfig: 统一的配置对象
        """
        mapping = {
            "rewrite": self.rewrite_llm,
            "research": self.research_llm,
            "embedding": self.embedding_llm,
            "preprocess": self.preprocessing_llm,
            "preprocessing": self.preprocessing_llm,
            "docling": self.docling_llm,
            "vlm": self.docling_vlm
        }
        
        normalized_name = name.lower().strip()
        if normalized_name not in mapping:
            raise ValueError(f"未知的 LLM 配置名称: '{name}'。可用选项: {list(mapping.keys())}")
            
        return mapping[normalized_name]


# --- 实例化 ---
try:
    settings = Settings()
except Exception as e:
    print(f"!!! 严重错误: 无法从 {ENV_FILE_PATH} 加载配置。")
    print(f"错误详情: {e}")
    if "validation error" in str(e).lower():
        print("提示: 请检查 .env 文件中是否包含所有必需的 API KEY 和 URL 配置。")
    raise e

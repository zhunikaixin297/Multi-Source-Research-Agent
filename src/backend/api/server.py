import uvicorn
import uuid
import os
import aiofiles
import asyncio
import time
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# 导入服务接口定义和请求模型 (注意别名，避免混淆)
from ..services.agent_service import AgentService, ReportRequest as ServiceReportRequest
from ..domain.models import DocumentSource
from ..services.ingestion_service import IngestionService
# 导入工厂方法
from ..services.factory import get_agent_service, get_workspace_ingestion_service
from ..infrastructure.repository.session_memory_manager import get_session_memory_manager
from ..infrastructure.repository.workspace_registry import get_workspace_registry
from ..core.config import settings
# 导入 API 层定义的 Schema
from .schemas import (
    ResearchRequest,
    ReviewRequest,
    WorkspaceCreateResponse,
    DocumentUploadResponse,
    ResearchStartResponse,
)

log = logging.getLogger(__name__)

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx"}


def sanitize_upload_filename(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    return Path(normalized).name.replace("\x00", "")


def ensure_allowed_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")


def resolve_cors_settings(origins, allow_credentials: bool):
    if isinstance(origins, str):
        if origins.strip() == "" or origins.strip() == "*":
            origins = ["*"]
        else:
            origins = [o.strip() for o in origins.split(",") if o.strip()]
    if not origins:
        origins = ["*"]

    if "*" in origins and allow_credentials:
        log.warning("CORS allow_credentials 与通配符 origins 冲突，已自动禁用 allow_credentials。")
        allow_credentials = False

    return origins, allow_credentials

# ==========================================
# 0. 定时清理后台任务
# ==========================================

async def auto_cleanup_workspaces():
    """
    后台任务：定期清理过期工作区。
    """
    manager = get_session_memory_manager(settings.session_rag)
    registry = get_workspace_registry(settings.session_rag.workspace_registry_db)
    
    while True:
        try:
            # 使用配置中的 TTL
            ttl = settings.session_rag.workspace_ttl
            expired_ids = registry.get_expired_workspaces(ttl)
            
            for workspace_id in expired_ids:
                log.info(f"正在自动清理过期工作区 (TTL 过期): {workspace_id}")
                manager.cleanup_workspace(workspace_id)
                registry.delete_workspace(workspace_id)
            
            # 每小时检查一次
            await asyncio.sleep(3600)
        except Exception as e:
            log.error(f"自动清理后台任务异常: {e}")
            await asyncio.sleep(60)

app = FastAPI(title="Research Agent API")

@app.on_event("startup")
async def startup_event():
    # 启动后台清理协程
    asyncio.create_task(auto_cleanup_workspaces())

cors_origins, cors_allow_credentials = resolve_cors_settings(
    settings.server.cors_allow_origins,
    settings.server.cors_allow_credentials,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. DeepResearch 服务接口
# ==========================================

@app.post("/api/tasks", response_model=ResearchStartResponse)
async def start_task(req: ResearchRequest):
    """
    启动任务接口
    """
    manager = get_session_memory_manager(settings.session_rag)
    manager.get_or_create_workspace(req.workspace_id)
    
    # 触碰工作区，更新活跃时间
    get_workspace_registry().touch_workspace(req.workspace_id)
    
    task_id = str(uuid.uuid4())
    return ResearchStartResponse(workspace_id=req.workspace_id, task_id=task_id)

@app.get("/api/tasks/{task_id}/stream")
async def stream_task(
    task_id: str,
    workspace_id: str,
    goal: str,
    service: AgentService = Depends(get_agent_service)
):
    """
    SSE 流式输出接口
    """
    # 触碰工作区
    get_workspace_registry().touch_workspace(workspace_id)

    # 构造 ServiceReportRequest 对象
    # 不再直接传 input_data，而是传封装好的 request 对象
    service_request = ServiceReportRequest(
        task_id=task_id,
        workspace_id=workspace_id,
        query=goal,
        action="start"
    )
    
    return StreamingResponse(
        service.generate_report(service_request),
        media_type="text/event-stream"
    )

@app.post("/api/tasks/review")
async def review_plan(
    req: ReviewRequest,
    service: AgentService = Depends(get_agent_service)
):
    """
    人工审核接口
    """
    if req.action not in ["approve", "revise"]:
        raise HTTPException(status_code=400, detail="Invalid action")

    # 触碰工作区
    get_workspace_registry().touch_workspace(req.workspace_id)

    # 构造 ServiceReportRequest 对象 (Resume 模式)
    service_request = ServiceReportRequest(
        task_id=req.task_id,
        workspace_id=req.workspace_id,
        action=req.action, 
        feedback=req.feedback,
        query=None # 恢复阶段通常不需要 query
    )

    return StreamingResponse(
        service.generate_report(service_request),
        media_type="text/event-stream"
    )

# ==========================================
# 2. 文档上传与解析接口
# ==========================================
# 限制最大文件大小
MAX_FILE_SIZE_MB = settings.server.max_file_size_mb
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# 辅助函数：异步保存文件
async def save_upload_file_async(upload_file: UploadFile, destination: str):
    file_size = 0
    try:
        async with aiofiles.open(destination, 'wb') as out_file:
            while content := await upload_file.read(1024 * 1024):  # 每次读取 1MB
                file_size += len(content)
                
                # Check: 如果超过最大限制
                if file_size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413, # Payload Too Large
                        detail=f"文件过大，超过限制 ({MAX_FILE_SIZE_MB}MB)"
                    )
                
                await out_file.write(content)
    except HTTPException:
        # 如果是大小超限触发的异常，抛出给上层
        # 在抛出前，必须删除这个只写了一半的垃圾文件
        if os.path.exists(destination):
            os.remove(destination)
        raise 
    except Exception as e:
        # 其他 IO 错误
        if os.path.exists(destination):
            os.remove(destination)
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

@app.post("/api/workspaces", response_model=WorkspaceCreateResponse)
async def create_workspace():
    workspace_id = str(uuid.uuid4())
    manager = get_session_memory_manager(settings.session_rag)
    manager.init_workspace_env(workspace_id)
    manager.workspace_documents_dir(workspace_id).mkdir(parents=True, exist_ok=True)
    
    # 初始化注册表记录
    get_workspace_registry().touch_workspace(workspace_id)
    
    return WorkspaceCreateResponse(workspace_id=workspace_id)


@app.post("/api/workspaces/{workspace_id}/documents", response_model=DocumentUploadResponse)
async def upload_workspace_document(
    workspace_id: str,
    file: UploadFile = File(...),
    ingestion_service: IngestionService = Depends(get_workspace_ingestion_service)
):
    """
    上传文件并触发解析流程，实时流式返回解析日志。
    """
    # 触碰工作区
    get_workspace_registry().touch_workspace(workspace_id)

    manager = get_session_memory_manager(settings.session_rag)
    manager.init_workspace_env(workspace_id)
    workspace_docs_dir = manager.workspace_documents_dir(workspace_id)
    workspace_docs_dir.mkdir(parents=True, exist_ok=True)
    current_count = manager.workspace_document_count(workspace_id)
    if current_count >= settings.session_rag.max_workspace_documents:
        raise HTTPException(
            status_code=400,
            detail=f"当前工作区文档数量已达上限 {settings.session_rag.max_workspace_documents}",
        )

    ensure_allowed_extension(file.filename)
    safe_name = sanitize_upload_filename(file.filename)
    safe_filename = f"{uuid.uuid4()}_{safe_name}"
    workspace_file_path = workspace_docs_dir / safe_filename
    
    # 直接保存到工作区目录
    await save_upload_file_async(file, str(workspace_file_path))

    source = DocumentSource(
        file_path=workspace_file_path,
        document_name=file.filename,
        document_id=str(uuid.uuid4()),
        metadata={"workspace_id": workspace_id},
    )

    chunk_count = await ingestion_service.pipeline_workspace_document(
        source=source,
        workspace_id=workspace_id,
    )
    return DocumentUploadResponse(
        workspace_id=workspace_id,
        file_name=file.filename,
        status="parsed",
        chunk_count=chunk_count,
    )

@app.delete("/api/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str):
    manager = get_session_memory_manager(settings.session_rag)
    manager.cleanup_workspace(workspace_id)
    
    # 从注册表中删除
    get_workspace_registry().delete_workspace(workspace_id)
    
    return {"status": "ok", "workspace_id": workspace_id}

if __name__ == "__main__":
    uvicorn.run(
        "src.backend.api.server:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,  # 默认关闭 reload，防止上传文件触发重启导致连接断开
    )

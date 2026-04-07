
import asyncio
import os
import sys
import shutil
import uuid
import pytest
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.backend.infrastructure.repository.session_memory_manager import get_session_memory_manager
from src.backend.services.factory import get_workspace_ingestion_service
from src.backend.domain.models import DocumentSource
from src.backend.core.config import settings

@pytest.mark.asyncio
async def test_workspace_lifecycle():
    if os.getenv("RUN_WORKSPACE_TESTS") != "1":
        pytest.skip("RUN_WORKSPACE_TESTS not set, skipping workspace lifecycle test")
    workspace_id = f"test_ws_{uuid.uuid4().hex[:8]}"
    manager = get_session_memory_manager(settings.session_rag)
    
    try:
        # 1. 初始化测试环境
        manager.init_workspace_env(workspace_id)
        assert manager.workspace_root_dir(workspace_id).exists()
        assert manager.workspace_images_dir(workspace_id).exists()
        assert manager.workspace_documents_dir(workspace_id).exists()
        
        # 2. 模拟上传文件
        sample_file = Path("tests/sample_documents/simple.pdf")
        if not sample_file.exists():
            pytest.skip("Sample document missing: tests/sample_documents/simple.pdf")
        dest_file = manager.workspace_documents_dir(workspace_id) / "simple.pdf"
        shutil.copy(sample_file, dest_file)
        
        assert manager.workspace_document_count(workspace_id) == 1
        
        # 3. 测试解析与入库
        ingestion_service = get_workspace_ingestion_service()
        source = DocumentSource(
            file_path=dest_file,
            document_name="simple.pdf",
            document_id=str(uuid.uuid4()),
            metadata={"workspace_id": workspace_id}
        )
        
        chunk_count = await ingestion_service.pipeline_workspace_document(
            source=source,
            workspace_id=workspace_id
        )
        
        assert chunk_count > 0
        
        # 4. 验证 Chroma 集合
        collection = manager.get_collection_if_exists(workspace_id)
        assert collection is not None
        assert collection.count() == chunk_count
        
    finally:
        # 5. 清理
        manager.cleanup_workspace(workspace_id)
        assert not manager.workspace_root_dir(workspace_id).exists()
        assert manager.get_collection_if_exists(workspace_id) is None

if __name__ == "__main__":
    asyncio.run(test_workspace_lifecycle())

import sqlite3
import time
import logging
from pathlib import Path
from typing import List, Optional
from ...core.config import BASE_DIR

log = logging.getLogger(__name__)

class WorkspaceRegistry:
    def __init__(self, db_path: str):
        # 如果是相对路径，则相对于 BASE_DIR
        path = Path(db_path)
        if not path.is_absolute():
            self.db_path = BASE_DIR / path
        else:
            self.db_path = path
            
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                # 开启 WAL 模式提高并发性能
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS workspaces (
                        workspace_id TEXT PRIMARY KEY,
                        last_accessed_at REAL NOT NULL,
                        created_at REAL NOT NULL
                    )
                """)
                conn.commit()
        except Exception as e:
            log.error(f"初始化工作区注册表数据库失败: {e}", exc_info=True)

    def touch_workspace(self, workspace_id: str):
        """更新或创建工作区访问记录"""
        now = time.time()
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("""
                    INSERT INTO workspaces (workspace_id, last_accessed_at, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(workspace_id) DO UPDATE SET last_accessed_at = excluded.last_accessed_at
                """, (workspace_id, now, now))
                conn.commit()
        except Exception as e:
            log.error(f"更新工作区访问时间失败 workspace_id={workspace_id}: {e}", exc_info=True)

    def get_expired_workspaces(self, ttl: int) -> List[str]:
        """获取已过期的工作区 ID 列表"""
        threshold = time.time() - ttl
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.execute(
                    "SELECT workspace_id FROM workspaces WHERE last_accessed_at < ?",
                    (threshold,)
                )
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            log.error(f"查询过期工作区失败: {e}", exc_info=True)
            return []

    def delete_workspace(self, workspace_id: str):
        """从注册表中删除工作区"""
        try:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("DELETE FROM workspaces WHERE workspace_id = ?", (workspace_id,))
                conn.commit()
        except Exception as e:
            log.error(f"从注册表删除工作区失败 workspace_id={workspace_id}: {e}", exc_info=True)

_registry: Optional[WorkspaceRegistry] = None

def get_workspace_registry(db_path: str = None) -> WorkspaceRegistry:
    global _registry
    if _registry is None:
        if db_path is None:
            from ...core.config import settings
            db_path = settings.session_rag.workspace_registry_db
        _registry = WorkspaceRegistry(db_path)
    return _registry

import json
import traceback
from typing import AsyncGenerator, Optional, Dict, Any, Sequence
from langgraph.types import Command
from langfuse.langchain import CallbackHandler

# 导入获取图的方法
from ..infrastructure.agents.orchestrator_agent import get_orchestrator_graph
from ..domain.interfaces import AgentService
from ..domain.models import ReportRequest
from ..infrastructure.langfuse.factory import init_langfuse_client
from ..core.config import settings


def _build_langfuse_trace_config(
    *,
    thread_id: str,
    workspace_id: str,
    action: str,
    trace_name: str,
    user_id: Optional[str] = None,
    extra_tags: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    tags = [
        "deepresearch-agent",
        f"workspace:{workspace_id}",
        f"thread:{thread_id}",
        f"action:{action}",
    ]

    if extra_tags:
        tags.extend(str(tag) for tag in extra_tags if tag)

    # 去重但保持顺序，避免 Langfuse tags 出现重复值。
    deduped_tags = list(dict.fromkeys(tags))

    metadata: Dict[str, Any] = {
        "langfuse_session_id": thread_id,
        "langfuse_tags": deduped_tags,
        "workspace_id": workspace_id,
        "thread_id": thread_id,
        "action": action,
        "trace_name": trace_name,
    }

    if user_id:
        metadata["langfuse_user_id"] = user_id

    return {
        "metadata": metadata,
        "tags": deduped_tags,
        "run_name": trace_name,
    }

class AgentServiceImpl(AgentService):
    def __init__(self):
        # 初始化图实例
        self.graph = get_orchestrator_graph()

    async def generate_report(self, request: ReportRequest) -> AsyncGenerator[str, None]:
        """
        实现 generate_report 接口。
        根据 request.action 决定是启动新任务还是恢复中断的任务。
        """
        thread_id = request.task_id
        workspace_id = request.workspace_id
        input_data = None
        resume_command = None

        # 1. 构建输入参数
        if request.action == "start":
            if not request.query:
                # 如果是开始请求，必须要有 query
                yield f"event: error\ndata: {json.dumps({'error': 'Start action requires a query'})}\n\n"
                return

            input_data = {
                "goal": request.query,
                "plan": [],
                "results": [],
                "loop_count": 0,
                "user_feedback": None,
                "reflection_feedback": None,
                "final_report": ""
            }
        
        elif request.action == "approve":
            resume_command = Command(resume={"action": "approve"})
            
        elif request.action == "revise":
            resume_command = Command(resume={"action": "revise", "feedback": request.feedback})

        # 2. 调用核心流生成逻辑
        # 复用原 event_stream_generator 的逻辑，但现在它是内部实现细节
        async for event in self._run_graph_stream(
            thread_id=thread_id,
            workspace_id=workspace_id,
            input_data=input_data,
            resume_command=resume_command,
        ):
            yield event

    async def _run_graph_stream(
        self, 
        thread_id: str,
        workspace_id: str,
        input_data: Optional[Dict[str, Any]] = None, 
        resume_command: Optional[Command] = None
    ) -> AsyncGenerator[str, None]:
        """
        内部方法：执行 Graph 并生成 SSE 格式的流。
        """
        
        # --- 🟢 修改开始：Langfuse 运行状态检测 ---
        langfuse = None
        callbacks = []
        trace_name = f"research-report-{thread_id}"
        
        try:
            # 1. 初始化客户端
            client = init_langfuse_client(
                public_key=settings.langfuse.public_key,
                secret_key=settings.langfuse.secret_key,
                base_url=settings.langfuse.base_url
            )
            
            # 2. 【关键步骤】执行连接检查
            # auth_check() 会发起一个轻量级请求验证凭证和连接
            # 如果连接被拒绝(Connection Refused)，这里会抛出异常或返回 False
            if client and client.auth_check():
                langfuse = client
                # 只有检查通过，才初始化 Handler 并加入回调列表
                langfuse_handler = CallbackHandler()
                callbacks.append(langfuse_handler)
            else:
                print("[Langfuse] ⚠️ Auth check failed or service down. Tracing skipped.")
                
        except Exception as e:
            # 捕获所有连接错误，防止应用崩溃
            print(f"[Langfuse] ⚠️ Connection check failed: {e}. Tracing skipped.")
            langfuse = None
        # --- 🟢 修改结束 ---

        config = {
            "configurable": {
                "workspace_id": workspace_id,
                "thread_id": thread_id,
            },
            "callbacks": callbacks, # 使用动态生成的 callbacks 列表
            **_build_langfuse_trace_config(
                thread_id=thread_id,
                workspace_id=workspace_id,
                action=request.action,
                trace_name=trace_name,
                extra_tags=[
                    f"goal:{request.query[:80]}" if request.query else None,
                ],
            ),
        }

        try:
            # 决定是启动新任务还是恢复中断
            if resume_command:
                async_generator = self.graph.astream_events(resume_command, config, version="v2")
            else:
                async_generator = self.graph.astream_events(input_data, config, version="v2")

            async for event in async_generator:
                kind = event["event"]
                
                # 1. 处理 LLM 流式输出 (Writer 节点)
                if kind == "on_chat_model_stream" and event["metadata"].get("langgraph_node") == "writer":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        yield f"event: report_token\ndata: {json.dumps({'token': chunk.content})}\n\n"

                # 2. 处理通用日志事件 (Agent Log)
                elif kind == "on_custom_event" and event["name"] == "agent_log":
                    yield f"event: log\ndata: {json.dumps(event['data'])}\n\n"
                
                # 3. 处理 Worker 进度事件
                elif kind == "on_custom_event" and event["name"] == "worker_progress":
                    yield f"event: progress\ndata: {json.dumps(event['data'])}\n\n"

            # 循环结束，检查最终状态或中断
            snapshot = await self.graph.aget_state(config)
            
            # Debug log
            print(f"Snapshot next: {snapshot.next}")
            
            if snapshot.next:
                # 尝试提取中断信息
                interrupt_value = None
                if snapshot.tasks:
                    for task in snapshot.tasks:
                        if task.interrupts:
                            interrupt_value = task.interrupts[0].value
                            break
                
                if interrupt_value:
                    yield f"event: interrupt\ndata: {json.dumps(interrupt_value)}\n\n"
                else:
                    # 异常中断状态：有 next 但无 interrupt 信息
                    yield f"event: log\ndata: {json.dumps({'message': '⚠️ 系统暂停，但未检测到明确的中断信号，请检查后端日志。'})}\n\n"
            else:
                # 任务完成
                final_report = snapshot.values.get("final_report")
                if final_report:
                    yield f"event: done\ndata: {json.dumps({'report': final_report})}\n\n"
                else:
                    pass

        except Exception as e:
            print(f"Stream Error: {e}")
            traceback.print_exc()
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        finally:
            # --- 🟢 修改：Flush 时增加判空逻辑 ---
            try:
                # 只有当 langfuse 实例成功创建且检查通过时，才尝试 flush
                if langfuse:
                    langfuse.flush()
            except Exception as e:
                # 如果日志发送失败（比如断网），只打印错误，不要影响业务的主流程
                print(f"[Langfuse] Flush failed: {e}")

import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
// 【注意】本地安装依赖后取消注释: npm install remark-gfm remark-math rehype-katex katex
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

import { FileText, CheckCircle, Edit3, Play, Loader, Clock, Terminal, Search, BookOpen, Upload, Trash2, RefreshCw, ChevronLeft, ChevronRight, File } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8002";

export default function App() {
  const [goal, setGoal] = useState("");
  const [workspaceId, setWorkspaceId] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState("idle"); 
  const [logs, setLogs] = useState([]);
  const [plan, setPlan] = useState(null);
  const [report, setReport] = useState("");
  const [feedback, setFeedback] = useState("");
  const [tasks, setTasks] = useState({});
  const [isLogsVisible, setIsLogsVisible] = useState(true);
  const [documents, setDocuments] = useState([]);
  
  // 上传相关状态
  const [uploadingCount, setUploadingCount] = useState(0);
  const fileInputRef = useRef(null);
  
  const eventSourceRef = useRef(null);
  const logsEndRef = useRef(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const addLog = (message) => {
    const time = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, { time, message }]);
  };

  const resetSession = async () => {
    if (status === "running") {
      if (!window.confirm("当前研究正在运行，确定要强行重置并清空所有内容吗？")) return;
      if (eventSourceRef.current) eventSourceRef.current.close();
    }

    if (workspaceId) {
      try {
        addLog(`🧹 正在物理销毁工作区: ${workspaceId}...`);
        await fetch(`${API_BASE}/api/workspaces/${workspaceId}`, { method: "DELETE" });
        addLog(`✅ 工作区资源已彻底清理。`);
      } catch (e) {
        addLog(`⚠️ 清理工作区失败: ${e.message}`);
      }
    }

    // 重置所有前端状态
    setWorkspaceId(null);
    setTaskId(null);
    setStatus("idle");
    setLogs([]);
    setPlan(null);
    setReport("");
    setTasks({});
    setFeedback("");
    setDocuments([]);
    setUploadingCount(0);
    addLog("✨ 会话已重置，可以开始新的研究。");
  };

  const ensureWorkspace = async () => {
    if (workspaceId) return workspaceId;
    const response = await fetch(`${API_BASE}/api/workspaces`, { method: "POST" });
    if (!response.ok) throw new Error("创建工作区失败");
    const data = await response.json();
    setWorkspaceId(data.workspace_id);
    addLog(`📦 已创建新工作区: ${data.workspace_id}`);
    return data.workspace_id;
  };

  const startResearch = async () => {
    if (status === "running") return;
    setStatus("running");
    setLogs([]);
    setReport("");
    setPlan(null);
    setTasks({}); 
    try {
      const wsId = await ensureWorkspace();
      const createTaskResp = await fetch(`${API_BASE}/api/tasks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal, workspace_id: wsId }),
      });
      if (!createTaskResp.ok) throw new Error("创建任务失败");
      const taskData = await createTaskResp.json();
      setTaskId(taskData.task_id);
      const url = `${API_BASE}/api/tasks/${taskData.task_id}/stream?workspace_id=${encodeURIComponent(wsId)}&goal=${encodeURIComponent(goal)}`;
      connectSSE(url);
    } catch (e) {
      addLog(`❌ 启动任务失败: ${e.message}`);
      setStatus("idle");
    }
  };

  const connectSSE = (url, isResume = false) => {
    if (eventSourceRef.current) eventSourceRef.current.close();
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onopen = () => {
        if (!isResume) addLog("连接成功，准备开始...");
    };

    es.addEventListener("log", (event) => {
      const data = JSON.parse(event.data);
      addLog(data.message);
    });

    es.addEventListener("progress", (event) => {
      const data = JSON.parse(event.data);
      setTasks(prev => ({
        ...prev,
        [data.task_id]: { ...prev[data.task_id], ...data }
      }));
      if (data.message) {
        addLog(data.message);
      }
    });

    es.addEventListener("interrupt", (event) => {
      const data = JSON.parse(event.data);
      setPlan(data.data);
      setStatus("waiting_review");
      // 此时流可以关闭，等待用户操作
      es.close();
    });

    es.addEventListener("report_token", (event) => {
      const data = JSON.parse(event.data);
      setReport(prev => prev + data.token);
    });

    es.addEventListener("done", () => {
      setStatus("completed");
      addLog("✅ 任务全部完成！");
      es.close();
    });

    // 【新增】监听后端明确发送的错误事件
    es.addEventListener("error", (event) => {
        // 这是自定义的 error 事件类型，不是 SSE 协议的 onerror
        const data = JSON.parse(event.data);
        console.error("Backend Error:", data);
        addLog(`❌ 后端错误: ${data.error}`);
        es.close();
        setStatus("idle");
    });

    // 原生 onerror 处理连接断开
    es.onerror = (err) => {
      if (es.readyState === EventSource.CLOSED) return;
      console.error("SSE Connection Error:", err);
      es.close();
      if (status !== "completed" && status !== "waiting_review") {
        addLog("⚠️ 连接异常断开 (请检查后端控制台)");
        setStatus("idle");
      }
    };
  };

  // --- 上传相关逻辑 ---
  const handleFileSelect = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadingCount(prev => prev + 1);
    addLog(`📤 开始上传文件: ${file.name}`);

    try {
      const wsId = await ensureWorkspace();
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${API_BASE}/api/workspaces/${wsId}/documents`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Upload failed");
      }
      const data = await response.json();
      addLog(`✅ 文件 ${file.name} 处理完成，入库分块: ${data.chunk_count}`);
      
      // 更新已上传文件列表
      setDocuments(prev => [...prev, { name: file.name, chunks: data.chunk_count, time: new Date().toLocaleTimeString() }]);
    } catch (e) {
      console.error(e);
      addLog(`❌ 上传/解析失败: ${e.message}`);
    } finally {
      setUploadingCount(prev => Math.max(0, prev - 1));
      // 清空 input 防止重复上传同个文件不触发 change
      e.target.value = ""; 
    }
  };

  const handleApprove = async () => {
    setStatus("running");
    setPlan(null); 
    await fetchAndStream("approve");
  };

  const handleRevise = async () => {
    if (!feedback.trim()) return alert("请输入修改意见");
    setStatus("running");
    setPlan(null);
    await fetchAndStream("revise", feedback);
  };

  const fetchAndStream = async (action, fb = null) => {
    try {
      const response = await fetch(`${API_BASE}/api/tasks/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task_id: taskId, workspace_id: workspaceId, action, feedback: fb }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); 

        lines.forEach(line => {
          if (line.startsWith("event: ")) {
            const match = line.match(/event: (.*)\n/);
            if (!match) return;
            const eventType = match[1];
            const dataMatch = line.match(/data: (.*)/);
            if (!dataMatch) return;
            const dataContent = dataMatch[1];
            
            if (eventType === "log") {
               const d = JSON.parse(dataContent);
               addLog(d.message);
            } else if (eventType === "progress") {
              const d = JSON.parse(dataContent);
              setTasks(prev => ({ ...prev, [d.task_id]: { ...prev[d.task_id], ...d } }));
              if (d.message) addLog(d.message);
            } else if (eventType === "report_token") {
              const d = JSON.parse(dataContent);
              setReport(prev => prev + d.token);
            } else if (eventType === "done") {
              setStatus("completed");
              addLog("✅ 任务全部完成！");
            } else if (eventType === "interrupt") {
               const d = JSON.parse(dataContent);
               setPlan(d.data);
               setStatus("waiting_review");
            } else if (eventType === "error") {
               const d = JSON.parse(dataContent);
               addLog(`❌ 后端错误: ${d.error}`);
            }
          }
        });
      }
    } catch (e) {
      console.error(e);
      addLog("❌ 恢复执行失败");
      setStatus("idle");
    }
  };

  const getCleanReport = (text) => {
    if (!text) return "";
    let clean = text.replace(/^```(markdown)?\s*/i, "");
    clean = clean.replace(/```\s*$/, "");
    return clean;
  };

  const sortedTasks = Object.values(tasks).sort((a, b) => (a.task_id || 0) - (b.task_id || 0));

  return (
    <div className="h-screen flex flex-col bg-gray-50 font-sans text-gray-800 overflow-hidden">
      <div className="max-w-screen-2xl mx-auto w-full flex flex-col h-full p-4 md:p-6">
        <header className="mb-4 text-center shrink-0">
          <h1 className="text-2xl md:text-3xl font-bold text-blue-600 flex items-center justify-center gap-2">
            <FileText className="w-8 h-8" /> DeepResearch Agent
          </h1>
          <p className="text-gray-500 text-sm mt-1">多源知识驱动的多智能体研究框架</p>
        </header>

        <div className="bg-white p-4 rounded-xl shadow-sm mb-4 border border-gray-100 shrink-0">
          <div className="flex flex-col md:flex-row gap-3 items-center">
            {/* 重置/新建按钮 */}
            <button
              onClick={resetSession}
              className="bg-red-50 text-red-600 px-3 py-3 rounded-xl hover:bg-red-100 border border-red-100 flex items-center justify-center gap-2 transition-all whitespace-nowrap text-sm group"
              title="重置会话并清空所有上传文档"
            >
              <RefreshCw className="w-4 h-4 group-hover:rotate-180 transition-transform duration-500" />
              重置
            </button>

            {/* 上传文件部分 */}
            <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                className="hidden" 
                accept=".pdf,.docx" 
            />
            <button
              onClick={handleFileSelect}
              disabled={status === "running"}
              className="bg-gray-50 text-gray-700 px-4 py-3 rounded-xl hover:bg-gray-100 disabled:bg-gray-50 disabled:text-gray-300 border border-gray-200 flex items-center justify-center gap-2 transition-all whitespace-nowrap text-sm relative"
              title="上传本地文档进行解析"
            >
              {uploadingCount > 0 ? <Loader className="animate-spin w-4 h-4" /> : <Upload className="w-4 h-4" />}
              上传文档
              {uploadingCount > 0 && (
                <span className="absolute -top-2 -right-2 bg-blue-500 text-white text-[10px] w-5 h-5 rounded-full flex items-center justify-center animate-pulse">
                  {uploadingCount}
                </span>
              )}
            </button>

            <div className="flex-1 relative">
                <input
                type="text"
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 text-base shadow-sm"
                placeholder="输入您的研究目标..."
                disabled={status === "running"}
                onKeyDown={(e) => e.key === 'Enter' && !status.match(/running|waiting/) && startResearch()}
                />
            </div>
            <button
              onClick={startResearch}
              disabled={status === "running" || status === "waiting_review"}
              className="bg-blue-600 text-white px-6 py-3 rounded-xl hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-bold text-base whitespace-nowrap shadow-sm transition-all"
            >
              {status === "running" ? <Loader className="animate-spin w-5 h-5" /> : <Play className="w-5 h-5" />}
              {status === "running" ? "研究中..." : "开始研究"}
            </button>
          </div>

          {/* 已上传文档列表展示 */}
          {documents.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2 animate-in slide-in-from-top-2 duration-300">
              {documents.map((doc, idx) => (
                <div key={idx} className="flex items-center gap-2 bg-blue-50/50 border border-blue-100 px-3 py-1.5 rounded-lg group hover:bg-blue-100/50 transition-colors">
                  <div className="bg-white p-1 rounded border border-blue-200 shadow-sm">
                    <File size={12} className="text-blue-500" />
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[11px] font-medium text-blue-800 truncate max-w-[150px]" title={doc.name}>
                      {doc.name}
                    </span>
                    <span className="text-[9px] text-blue-400 leading-none">
                      {doc.chunks} chunks • {doc.time}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="flex gap-4 flex-1 min-h-0 pb-2 relative">
          {/* 左侧：日志区 (可收起) */}
          <div className={`${isLogsVisible ? 'w-1/4' : 'w-10'} flex flex-col h-full min-h-0 transition-all duration-300 ease-in-out relative group`}>
             <button 
                onClick={() => setIsLogsVisible(!isLogsVisible)}
                className="absolute -right-3 top-1/2 -translate-y-1/2 z-20 bg-white border border-gray-200 rounded-full p-1 shadow-sm hover:bg-gray-50 text-gray-400 hover:text-blue-500 transition-all opacity-0 group-hover:opacity-100"
                title={isLogsVisible ? "收起日志" : "展开日志"}
             >
                {isLogsVisible ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
             </button>

             <div className={`bg-gray-900 text-green-400 rounded-xl shadow-md h-full overflow-hidden border border-gray-800 flex flex-col font-mono transition-all duration-300 ${isLogsVisible ? 'p-4 opacity-100' : 'p-0 opacity-0 pointer-events-none'}`}>
              <h2 className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-3 flex items-center gap-2 border-b border-gray-700 pb-2 shrink-0">
                <Terminal size={12} /> 执行日志
              </h2>
              <div className="flex-1 overflow-y-auto space-y-2 pr-2 custom-scrollbar scroll-smooth text-xs">
                {logs.length === 0 && <div className="text-gray-600 italic text-center mt-10">...</div>}
                {logs.map((log, i) => (
                  <div key={i} className="flex gap-2">
                    <span className="text-gray-500 text-[10px] min-w-[45px] shrink-0">{log.time}</span>
                    <span className={`
                        ${log.message.includes("✅") ? "text-green-300 font-bold" : ""}
                        ${log.message.includes("❌") ? "text-red-400" : ""}
                        ${log.message.includes("上传") || log.message.includes("解析") ? "text-purple-300" : "text-gray-300"}
                        ${log.message.includes("不足") ? "text-yellow-300" : ""}
                        break-all
                    `}>
                        {log.message}
                    </span>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            </div>

            {!isLogsVisible && (
                <div 
                    onClick={() => setIsLogsVisible(true)}
                    className="h-full w-full bg-gray-100 rounded-xl border border-dashed border-gray-300 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-200 transition-colors group/mini"
                >
                    <Terminal size={16} className="text-gray-400 group-hover/mini:text-blue-500" />
                    <span className="[writing-mode:vertical-lr] text-[10px] text-gray-400 mt-4 font-bold uppercase tracking-widest group-hover/mini:text-blue-500">执行日志</span>
                </div>
            )}
          </div>
          
          {/* 中间：任务进度 */}
          <div className="flex-1 flex flex-col h-full min-h-0 transition-all duration-300">
            <div className="bg-white p-4 rounded-xl shadow-md h-full overflow-hidden border border-gray-100 flex flex-col">
              <h2 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3 flex items-center gap-2 border-b pb-2 sticky top-0 bg-white z-10 shrink-0">
                <Search size={14} /> 实时研究进展
              </h2>
              <div className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar">
                {sortedTasks.length === 0 && (
                  <div className="text-gray-400 text-center mt-20 text-xs">暂无任务，请开始研究...</div>
                )}
                {sortedTasks.map((task) => (
                  <div key={task.task_id} className={`p-3 rounded-lg border transition-all ${task.status === 'completed' ? 'border-green-100 bg-green-50/50' : 'border-blue-100 bg-blue-50/50'}`}>
                    <div className="flex justify-between items-start mb-1.5">
                      <h4 className="font-bold text-gray-800 text-xs flex items-center gap-2">
                        <span className="bg-white border border-gray-200 text-gray-500 text-[10px] px-1.5 py-0.5 rounded">#{task.task_id}</span>
                        {task.title}
                      </h4>
                      {task.status === 'researching' ? (
                        <Loader size={12} className="animate-spin text-blue-500" />
                      ) : (
                        <CheckCircle size={12} className="text-green-500" />
                      )}
                    </div>
                    
                    {task.status === 'researching' && (
                      <p className="text-[10px] text-blue-600 animate-pulse">🔍 {task.message.replace("正在研究", "").replace("任务...", "")} (研究中...)</p>
                    )}
                    
                    {task.status === 'completed' && (
                      <div className="mt-1.5 text-[11px] text-gray-600 bg-white p-2 rounded border border-green-50 leading-relaxed">
                        <strong className="block text-green-700 mb-0.5 text-[10px] uppercase tracking-tighter">💡 研究结论:</strong>
                        {task.summary}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* 右侧：最终报告 */}
          <div className="flex-1 flex flex-col h-full min-h-0 transition-all duration-300">
            <div className="bg-white p-5 rounded-xl shadow-md h-full overflow-hidden border border-gray-100 relative flex flex-col">
              <h2 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3 flex items-center gap-2 border-b pb-2 sticky top-0 bg-white z-10 shrink-0">
                <BookOpen size={14} /> 最终报告
                {status === "running" && report && <span className="text-[10px] normal-case font-normal bg-blue-100 text-blue-600 px-2 py-0.5 rounded-full ml-2 animate-pulse">Writing...</span>}
              </h2>
              
              <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
                {report ? (
                  <div className="markdown-content max-w-none break-words text-sm leading-relaxed prose prose-blue prose-sm">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {getCleanReport(report)}
                    </ReactMarkdown>
                    {status === "running" && <span className="inline-block w-1.5 h-3 bg-blue-500 ml-1 animate-pulse align-middle"></span>}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-gray-300 space-y-3">
                    <FileText size={40} className="opacity-10" />
                    <p className="text-xs font-medium text-gray-400">最终报告将在此处生成</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* 模态框：人工审核计划 */}
        {status === "waiting_review" && plan && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-200">
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl max-h-[85vh] flex flex-col overflow-hidden border border-gray-200">
              <div className="p-6 border-b bg-gray-50 flex justify-between items-center">
                <div>
                    <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                        <CheckCircle className="text-blue-600" /> 审核研究大纲
                    </h3>
                    <p className="text-gray-500 text-sm mt-1">AI 已规划 {plan.length} 个子任务，请确认是否执行。</p>
                </div>
                <div className="text-xs bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full font-medium">
                    需人工确认
                </div>
              </div>
              
              <div className="p-6 overflow-y-auto flex-1 space-y-4 bg-white">
                {plan.map((task) => (
                  <div key={task.id} className="group bg-white p-5 rounded-xl border border-gray-200 hover:border-blue-300 hover:shadow-md transition-all">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-bold text-blue-600 text-lg flex items-center gap-2">
                        <span className="bg-blue-100 text-blue-700 text-xs px-2 py-0.5 rounded">Task {task.id}</span>
                        {task.title}
                      </h4>
                    </div>
                    <div className="space-y-1 pl-1">
                        <p className="text-sm text-gray-700"><strong className="text-gray-900">目标:</strong> {task.intent}</p>
                        <p className="text-sm text-gray-500 flex items-center gap-1">
                            <Clock size={12} /> 
                            Query: {task.query}
                        </p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="p-6 border-t bg-gray-50 space-y-4">
                <div className="flex gap-3">
                  <input
                    type="text"
                    placeholder="如果大纲不准确，请在此输入修改建议..."
                    value={feedback}
                    onChange={(e) => setFeedback(e.target.value)}
                    className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:outline-none shadow-sm"
                  />
                </div>
                <div className="flex gap-3 justify-end">
                  <button
                    onClick={handleRevise}
                    className="px-6 py-2.5 text-gray-700 bg-white border border-gray-300 hover:bg-gray-100 rounded-xl font-medium flex items-center gap-2 transition-colors shadow-sm"
                  >
                    <Edit3 size={18} />
                    提出修改
                  </button>
                  <button
                    onClick={handleApprove}
                    className="px-8 py-2.5 text-white bg-blue-600 hover:bg-blue-700 rounded-xl font-bold flex items-center gap-2 shadow-md transition-all active:scale-95"
                  >
                    <CheckCircle size={18} />
                    批准并执行
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

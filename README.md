# 多源知识驱动的多智能体研究框架

[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61dafb.svg)](https://reactjs.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-purple.svg)](https://langchain-ai.github.io/langgraph)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 项目简介

本项目是一个面向复杂业务场景的自动化调研系统。通过 **LangGraph** 状态机编排多智能体协作流，并核心引入 **MCP (Model Context Protocol)** 协议实现推理编排层与知识检索层的物理隔离，支持百万级 Token 背景下的高效率、可观测深度研究。

***

## ✨ 核心机制 (Core Mechanisms)

### 🎯 多智能体协作流 (Multi-Agent Workflow)

基于图编排，包含 Planner（规划）、Worker（检索分析）、Reporter（撰写）与 Reflector（反思）节点。支持在研究计划生成后引入人机交互 (Human-in-the-loop) 进行干预与确认。

### 🧠 三路知识路由 (Multi-Source Knowledge Routing)

Worker Agent 根据任务意图，通过语义路由动态选择知识源：

- **企业全局记忆 (MCP-based Global RAG)**：通过自研 MCP 接口跨进程接入独立 RAG 服务集群（如 [MODULAR-RAG-MCP-SERVER](https://github.com/jerry-ai-dev/MODULAR-RAG-MCP-SERVER)），实现海量历史文档的标准化检索。
- **会话即时记忆 (Session-level Local RAG)**：针对当前上传文档，基于 ChromaDB 构建轻量化索引，结合 **Docling** 完成高精度多模态解析（表格、公式、图像语义补全）。
- **动态公域检索 (Real-time Web Search)**：集成 Tavily/DuckDuckGo 进行时效性信息补充，降低闭门推理导致的事实滞后风险。

### 📦 会话级资源隔离 (Session Lifecycle Management)

基于 `task_id` 维护独立工作空间（Workspace），确保不同调研任务的上下文、索引与物理存储彻底阻断。通过生命周期钩子在任务结束后执行资源原子化销毁，降低长连接场景下的状态污染与脏数据残留风险。

***

## 🏗️ 架构设计

### 架构演进：微服务化与物理隔离

针对原生 Agent 架构中检索模块高度耦合的问题，系统完成了检索链路的微服务化改造。通过实现 **MCP Client** 与标准工具协议，Agent 可以以 Tool 形式调用外部独立 RAG 服务。该改造将计算层（Agent 推理编排）与存储层（检索与索引）进行物理解耦，并显著提升了服务治理与水平扩展能力。

### 系统架构图

```
┌──────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ 研究任务启动  │  │  实时进度展示 │  │  结果查看     │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌───────────────────────────────────────────────────────────────┐
│                      Backend API (FastAPI)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │/api/research/│  │/api/workspaces/│ │/api/documents/│       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼──────────────────┼──────────────────┼───────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                Multi-Agent Layer (LangGraph)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │协调器     │  │ 规划器   │  │ 执行器    │  │ 反思器   │     │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘     │
│        │            │            │            │             │
│        └────────────┴────────────┴────────────┘             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   ChromaDB   │  │  MCP Client  │  │  Web Search  │       │
│  │ (Session RAG)│  │ (Global RAG) │  │ (Tavily/DDG) │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

***

## 🚀 快速开始

### 环境要求

在开始之前，请确保您的系统已安装以下依赖：

- **Python**: 3.11+
- **Node.js**: 18+
- **Docker**: 20.10+
- **UV**: 高性能Python包管理器
- **Ollama**: 本地大模型推理引擎（用于embedding模型）

### 1. 克隆项目和安装依赖

```bash
git clone https://github.com/yjryjrhehe/deepresearch-agent.git
cd deepresearch-agent

# 首先安装 uv
# pip install uv
uv sync
```

### 2. 配置环境变量

项目需要在项目根目录配置 `.env` 文件。先复制并重命名`.env.example` 模板文件：

```bash
cp .env.example .env
```

然后编辑 `.env` 文件，填入您的API密钥：

```env
# 智谱GLM API Key
GLM_API_KEY="your_glm_api_key"

# 阿里通义千问API Key
QWEN_API_KEY="your_qwen_api_key"
```

**注意**：请将示例API密钥替换为您自己申请的真实密钥。

**API密钥获取方式：**

- **智谱GLM API Key**: 访问 [智谱AI开放平台](https://open.bigmodel.cn/) 注册并获取API密钥
- **阿里通义千问API Key**: 访问 [DashScope控制台](https://dashscope.console.aliyun.com/) 注册并获取API密钥

### 3. 配置Ollama并拉取Embedding模型

项目使用Ollama作为Embedding模型的本地推理引擎。首先安装Ollama并拉取指定模型：

**安装Ollama:**

```bash
# Windows
# 从 https://ollama.ai 下载并安装

# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

**拉取Embedding模型:**

```bash
ollama pull qwen3-embedding:4b
```

该模型用于将文档和查询转换为向量表示，是RAG系统的核心组件。如果您的硬件资源有限，可以选择更小的模型，但需要在配置文件中修改相应设置。

### 4. 运行项目启动脚本

项目提供了统一的启动脚本 `run.py`，该脚本会自动：

- 执行环境预检（包含可选模型兼容检查）
- 检查并安装前端依赖（npm install）
- 启动LiteLLM代理服务（端口4000）
- 启动FastAPI后端服务（端口8002）
- 启动React前端应用（端口5173）

```bash
# 运行启动脚本
python run.py
```

**启动脚本功能说明：**

1. **环境预检**: 检查环境与依赖，安装前端依赖
2. **LiteLLM代理**: 启动统一模型代理服务，端口4000
3. **FastAPI后端**: 启动图状态机 API 后端服务，端口8002
4. **React前端**: 启动可视化交互界面，端口5173


### 5. （可选）启动Langfuse追踪服务

如果需要使用Langfuse进行提示词管理、模型输入输出追踪和评估，启动可选的Langfuse服务：

项目需要在项目根目录配置 `.env` 文件。先复制并重命名`.env.example` 模板文件：

```bash
cd langfuse
cp .env.example .env
```

然后编辑 `.env` 文件，进行以下设置：

```env
# --- 1. 数据库密码 ---
# 设置 PostgreSQL 密码
POSTGRES_PASSWORD=xxx

# 设置 ClickHouse 密码
CLICKHOUSE_PASSWORD=xxx

# 设置 Minio 密码
MINIO_ROOT_PASSWORD=xxx

# 设置 Redis 密码
REDIS_AUTH=xxx

# --- 2. Langfuse 核心安全密钥 ---
# 必须是一个 64 位的十六进制字符串 (32 字节)。
# 在 Linux/macOS 上运行: openssl rand -hex 32
# 在 Windows 上运行：python -c "import secrets; print(secrets.token_hex(32))"
ENCRYPTION_KEY=90e2abbf1940d5c028627e86398883703437be3a4fd1xxxxxx

# 用于会话签名的随机字符串
NEXTAUTH_SECRET=4f65e01508625195fbdacb56bbcb6337fc3b0ac238d52xxxxx

# 用于内部哈希的随机字符串
SALT=ea92f2469f59310851d1b9fb30c33915c5c30ab3d0xxxxx
```

**注意**：相关密码设置需要包含数字、大小写子母和字符。

```bash
docker-compose up -d
```

**Langfuse服务说明：**

- **langfuse-web**: Web界面（端口3000）
- **postgres**: 数据库（端口6432）
- **redis**: 缓存（端口6379）
- **clickhouse**: 分析数据库（端口7125）
- **minio**: 对象存储（端口9090）

如果不使用Langfuse，系统将使用代码中写死的提示词模板。

### 6. 访问应用

当所有服务启动完成后，访问以下地址：

- **前端应用**: <http://localhost:5173>
- **API文档**: <http://localhost:8002/docs>
- **Langfuse Web** (可选): <http://localhost:3000>

### 完整启动顺序

1. `git clone` 项目
2. `uv sync` 配置环境
3. 配置 `.env` 环境变量信息（填入API密钥）
4. 安装 Ollama 并执行 `ollama pull qwen3-embedding:4b`
5. 运行 `python run.py`（一键启动前后端及 LiteLLM 服务）
6. 运行 `langfuse/docker-compose up -d`（可选）

***

## 📋 配置说明

### LiteLLM模型配置

在 `config.yaml` 中配置你的LLM模型。本项目支持的模型包括云端模型和本地模型：

```yaml
model_list:
  # 智谱GLM云端模型
  - model_name: "glm-4-flashx"
    litellm_params:
      model: "openai/glm-4-flashx-250414"
      api_base: "https://open.bigmodel.cn/api/paas/v4/"
      api_key: "os.environ/GLM_API_KEY"

  - model_name: "glm-4v-plus"
    litellm_params:
      model: "openai/glm-4v-plus-0111"
      api_base: "https://open.bigmodel.cn/api/paas/v4/"
      api_key: "os.environ/GLM_API_KEY"

  # 阿里通义千问云端模型
  - model_name: "qwen3-max"
    litellm_params:
      model: "openai/qwen3-max"
      api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
      api_key: "os.environ/QWEN_API_KEY"

  # 本地Ollama模型（用于对话）
  - model_name: "qwen3-4b-local"
    litellm_params:
      model: "ollama/qwen3:4b"
      api_base: "http://localhost:11434"

  # 本地Ollama模型（用于embedding）
  - model_name: "qwen3-embedding-4b-local"
    litellm_params:
      model: "ollama/qwen3-embedding:4b"
      api_base: "http://localhost:11434"
      drop_params: True

server_settings:
  port: 4000
  host: "0.0.0.0"
```

### 环境变量配置

在 `.env` 文件中配置关键参数。以下是完整的配置项：

```env
# ====================
# API密钥配置
# ====================
GLM_API_KEY="your_glm_api_key"          # 智谱AI API密钥
QWEN_API_KEY="your_qwen_api_key"        # 阿里云API密钥

# ====================
# 日志配置
# ====================
LOG_LEVEL="INFO"

# ====================
# LiteLLM代理配置
# ====================
LITELLM_PROXY_URL="http://127.0.0.1:4000"

# ====================
# Docling文档解析配置
# ====================
DOCLING_VLM_API_KEY="xx"
DOCLING_VLM_BASE_URL="http://127.0.0.1:4000"
DOCLING_VLM_MODEL="glm-4v-plus"
DOCLING_VLM_MAX_CONCURRENCY="5"

DOCLING_LLM_API_KEY="xx"
DOCLING_LLM_BASE_URL="http://127.0.0.1:4000"
DOCLING_LLM_MODEL="glm-4-flashx"
DOCLING_LLM_MAX_CONCURRENCY="30"

# ====================
# 文本分块配置
# ====================
MAX_CHUNK_TOKENS=1024
ENCODING_NAME="cl100k_base"
CHUNK_OVERLAP_TOKENS=100

# ====================
# 预处理LLM配置
# ====================
PREPROCESSING_LLM_API_KEY="xx"
PREPROCESSING_LLM_BASE_URL="http://127.0.0.1:4000"
PREPROCESSING_LLM_MODEL="glm-4-flashx"
PREPROCESSING_LLM_MAX_CONCURRENCY=30

# ====================
# Embedding模型配置
# ====================
EMBEDDING_LLM_API_KEY="xx"
EMBEDDING_LLM_BASE_URL="http://127.0.0.1:4000"
EMBEDDING_LLM_MODEL="qwen3-embedding-4b-local"  # 使用Ollama本地模型
EMBEDDING_LLM_DIMENSION=2560
EMBEDDING_LLM_MAX_CONCURRENCY=5

# ====================
# 查询重写LLM配置
# ====================
REWRITE_LLM_API_KEY="xx"
REWRITE_LLM_BASE_URL="http://127.0.0.1:4000"
REWRITE_LLM_MODEL="glm-4-flashx"
REWRITE_LLM_MAX_CONCURRENCY=30

# ====================
# 研究主模型配置
# ====================
RESEARCH_LLM_API_KEY="xx"
RESEARCH_LLM_BASE_URL="http://127.0.0.1:4000"
RESEARCH_LLM_MODEL="qwen3-max"  # 主要研究任务使用的模型
RESEARCH_LLM_MAX_CONCURRENCY=30

# ====================
# Langfuse追踪配置
# ====================
LANGFUSE_SECRET_KEY="sk-lf-xxx"  # 替换为你的密钥
LANGFUSE_PUBLIC_KEY="pk-lf-xxx"  # 替换为你的公钥
LANGFUSE_BASE_URL="http://localhost:3000"
```

### Embedding 模型配置说明

**Embedding模型（向量化引擎）:**

项目使用Ollama作为Embedding模型的本地推理引擎。默认配置使用 `qwen3-embedding:4b` 模型，该模型：

- 参数量：4B
- 向量维度：2560
- 优点：本地运行，隐私安全，成本低
- 推理地址：<http://localhost:11434>

如需更换其他embedding模型，请执行以下步骤：

1. **拉取新模型**（以nomic-embed-text为例）:

```bash
ollama pull nomic-embed-text
```

1. **修改`config.yaml`**:

```yaml
- model_name: "nomic-embedding-local"
  litellm_params:
    model: "ollama/nomic-embed-text"
    api_base: "http://localhost:11434"
    drop_params: True
```

1. **修改`.env`文件中的模型名称**:

```env
EMBEDDING_LLM_MODEL="nomic-embedding-local"
```

**硬件建议**: 4B模型建议至少4GB显存，推荐8GB以上。如果显存不足，可以选择1B参数的模型。

## 📖 使用指南

### 1. 文档上传与解析

访问前端应用 <http://localhost:5173，点击"上传文档"按钮，选择需要研究的PDF或Word文档。>

<div align="center">
  <img src="assets/parse-documents.png" width="800" />
  <br>
  <i>图 2：上传文档与解析</i>
</div>

**上传流程说明：**

当文档上传后，系统会自动进行以下处理：

1. **文档解析阶段**
   - 识别文档类型和结构
   - 提取文本内容
   - 识别表格、公式、图片等复杂元素
2. **VLM增强阶段**（可选）
   - 使用视觉语言模型识别图片中的文字和图表
   - 对复杂表格进行结构化处理
   - 提取并转换数学公式
3. **智能分块阶段**
   - 基于语义和结构将文档切分成合适大小的文本块
   - 每个文本块约1024个token
   - 保持块与块之间的上下文关联
4. **向量化与索引阶段**
   - 使用embedding模型将文本块转换为向量
   - 向量维度：2560
   - 将向量和元数据存储到Chroma

**预期耗时：** 文档解析时间取决于文档大小和复杂度。（实测，会话文档上传 30+ 页 PDF 不超过 1 分钟。）

### 2. 创建研究任务

<div align="center">
  <img src="assets/research-init.png" width="800" />
  <br>
  <i>图 3：创建研究任务</i>
</div>

在主界面的输入框中输入您的研究目标或问题。有效的输入应该：

- **明确具体**：例如"分析Transformer架构在大规模语言模型中的应用"而不是"AI相关"
- **包含上下文**：说明您希望从哪些角度进行分析
- **限定范围**：可以指定时间范围、行业领域等

**示例输入：**

- "分析2024年生成式AI在金融行业的应用现状、主要挑战和未来趋势"
- "研究量子计算在优化问题中的优势、当前技术瓶颈及商业化前景"
- "对比分析GPT、BERT、T5等预训练模型在文本摘要任务上的性能差异"

### 3. 智能体协作流程

系统会启动多智能体协作流程，包括四个智能体的有序协作：

#### 阶段1：规划（Orchestrator + Planner）

协调器首先分析用户输入，生成研究计划，然后由规划器智能体详细展开研究大纲。

<div align="center">
  <img src="assets/research-plan.png" width="800" />
  <br>
  <i>图 4：智能体生成研究大纲</i>
</div>

**流程：**

1. 协调器接收研究任务，分解为子问题
2. 规划器基于文档库和子问题生成详细的研究大纲
3. 研究大纲包括：
   - 主要研究方向和子主题
   - 每个方向的预期产出
   - 研究深度和广度建议
4. 大纲展示给用户，等待审核

**用户交互：**

- 可以选择"同意"开始执行
- 可以选择"修改"并提供反馈，规划器将基于反馈调整大纲

#### 阶段2：执行（Worker）

执行器智能体根据审核通过的研究大纲，对每个子主题进行深度研究。

<div align="center">
  <img src="assets/researching.png" width="800" />
  <br>
  <i>图 5：智能体研究过程</i>
</div>

**执行流程：**

1. 针对每个研究方向，执行器会：
   - 查询相关文档内容（混合检索：关键词+向量+重排序）
   - 分析文档片段
   - 提取关键信息和证据
   - 生成初步分析结果
2. 研究过程中的实时反馈：
   - SSE流式输出当前进度和发现
   - 显示正在分析的具体文档
   - 实时展示生成的段落内容
3. 检索优化：
   - 首先基于关键词进行初步检索
   - 使用embedding模型进行语义匹配
   - 对候选文档执行相关性重排（重排服务不可用时自动降级）
   - 提取Top-K相关内容进行分析

**预期耗时：** 每个研究方向需要1-5分钟，取决于文档库大小和研究深度。

#### 阶段3：反思（Reflector）

反思器智能体评估研究结果的质量和完整性。

<div align="center">
  <img src="assets/research-reflect.png" width="800" />
  <br>
  <i>图 6：智能体反思阶段</i>
</div>

**评估维度：**

1. **内容完整性**：是否覆盖了研究大纲的所有要点
2. **证据充分性**：是否提供了足够的文档证据支持结论
3. **逻辑一致性**：论证过程是否逻辑清晰、无矛盾
4. **深度分析**：是否进行了深度分析而非浅层描述

**优化策略：**

- 如果发现内容不足，反思器会标记不足的领域
- 触发执行器对标记领域进行补充研究
- 最多可迭代3轮优化

#### 阶段4：报告生成（Reporter）

报告生成器将所有研究结果汇总成结构化的研究报告。

<div align="center">
  <img src="assets/reporting.png" width="800" />
  <br>
  <i>图 7：报告生成阶段</i>
</div>

**报告结构：**

- 执行摘要：核心发现和结论
- 研究背景：问题定义和研究范围
- 详细分析：每个子主题的深度研究结果
- 数据支撑：引用文档中的关键证据
- 总结与建议：总体结论和未来方向
- 参考文献：引用文档清单

### 4. 实时进度监控

系统提供全方位的进度追踪和状态可视化：

**前端可视化组件：**

- 📊 **进度条**：显示当前阶段和整体进度百分比
- 📝 **实时日志**：滚动显示详细的执行日志
- 🔍 **当前状态**：展示正在执行的智能体和具体操作
- 📄 **文档追踪**：显示正在分析的文档名称
- 💬 **流式输出**：实时显示生成的文本内容

**状态指示器：**

- 🟢 绿色：正常执行
- 🟡 黄色：等待用户输入或优化中
- 🔴 红色：遇到错误或异常
- ⚪ 灰色：等待中

**交互控制：**

- 支持随时查看历史进度
- 支持中断当前研究（Ctrl+C）
- 支持恢复中断的研究任务

### 5. 查看研究结果

**前端结果展示：**

1. **主报告视图**
   - Markdown格式渲染，支持LaTeX数学公式
2. **引用文档导航**
   - 点击文档引用可直接跳转到相关段落
   - 查看文档来源和上下文
   - 支持多文档交叉引用
3. **性能指标**
   - 文档处理数量
   - 检索相关性评分
   - 智能体执行时间统计

***

## 🔍 Langfuse 提示词管理与追踪

Langfuse是一个开源的LLM应用可观测性平台，本项目集成了Langfuse用于提示词管理、模型输入输出追踪和效果评估。

### Langfuse 功能说明

**1. 提示词管理**

- 集中管理所有智能体的提示词模板
- 版本控制：记录提示词的每次修改历史
- A/B测试：对比不同提示词版本的效果
- 动态更新：无需重启服务即可更新提示词

**2. 模型追踪**

- 记录每个LLM调用的输入和输出
- 追踪token消耗和延迟
- 监控模型性能和成本
- 识别异常调用和错误

**3. 效果评估**

- 人工评估：对研究结果进行评分和反馈
- 自动评估：基于预定义指标自动评分
- 对比分析：不同模型、不同提示词的效果对比
- 趋势分析：长期性能变化趋势

### 使用 Langfuse

见 [Langfuse Documentation - Langfuse](https://langfuse.com/docs)。

***

## 🛠️ 项目结构

```
deepresearch-agent/
├── src/
│   ├── backend/
│   │   ├── api/                    # API层
│   │   │   ├── server.py          # FastAPI入口
│   │   │   └── schemas.py         # 数据模型
│   │   ├── core/                  # 核心层
│   │   │   ├── config.py          # 配置管理
│   │   │   └── logging.py         # 日志配置
│   │   ├── domain/                # 领域层
│   │   │   ├── models.py          # 业务模型
│   │   │   └── interfaces.py      # 接口定义
│   │   ├── infrastructure/        # 基础设施层
│   │   │   ├── agents/            # 多智能体系统
│   │   │   ├── parse/             # 文档解析
│   │   │   ├── repository/        # 数据存储和检索
│   │   │   ├── llm/               # LLM集成
│   │   │   └── langfuse/          # 监控追踪
│   │   └── services/              # 服务层
│   │       ├── agent_service.py   # 深度研究智能体服务
│   │       ├── ingestion_service.py # 数据摄入和解析服务
│   │       └── factory.py
│   └── frontend/
│       ├── src/
│       │   ├── main.jsx           # 应用入口
│       │   ├── App.jsx            # 主组件
│       │   ├── index.css          # 全局样式
│       │   └── App.css            # 组件样式
│       ├── package.json
│       └── vite.config.js
├── models/                        # 可选模型缓存目录（按需生成）
|—— langfuse/                      # langfuse部署文件
|	└── .env					 # 环境变量
|	└── docker-compose.yml         # 容器编排
├── data/                          # 核心数据目录 (工作区、向量库、数据库)
├── docker-compose.yml             # 可选扩展服务编排
├── config.yaml                    # LiteLLM配置
├── pyproject.toml                 # Python依赖
├── .env                           # 环境变量
└── README.md                      # 项目文档
```

***

## 🚀 后续扩展方向 (Roadmap & Future Work)

当前架构已完成核心数据链路的打通，未来计划在以下维度进行演进：

### 工作空间机制 (Workspace-based Tasks)

- **现状**：一次会话对应单一报告任务，任务结束后文档库即销毁。
- **演进**：引入 SQLite 维护二维映射 (Workspace ID 与 Task ID)。允许在同一个工作空间内保留文档池，支持基于同一批文档并发生成多份不同视角的研究报告。

### 多模态联合推理 (Multimodal VLM Integration)

- **现状**：Docling 解析时已提取原始图表，并将 `image_id` 作为元数据 (Metadata) 注入文本块。
- **演进**：引入视觉大模型 (Vision LLM)。当向量检索命中包含 `image_id` 的文本块时，自动回溯读取物理图片，使得 Agent 能够理解复杂的财务报表和数据图。

### Graph RAG 与结构化抽取

- **现状**：采用标准的密集向量 (Dense Vector) 检索，对跨文档的多跳逻辑推理能力有限。
- **演进**：在长时记忆 (MCP Server) 的后台流水线中引入命名实体识别 (NER) 与关系抽取，构建知识图谱，应对复杂的供应链或股权穿透研究。

### 多轮交互优化 (Iterative Refinement)

- **现状**：仅在规划阶段 (Plan) 允许人工介入，最终报告一键生成。
- **演进**：将状态机从严格的 DAG（有向无环图）升级为支持局部循环的架构。引入独立的 Revisor Agent，支持用户对最终报告提出针对性修改意见（如“重写第三章节”），并通过细粒度的状态管理控制 Token 开销。

***

## 📄 许可证

本项目基于MIT许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情

***

## 🙏 致谢与参考

本项目的系统性演进受益于以下开源生态：

- **架构启发**：感谢 [DeepResearch](https://github.com/yjryjrhehe/deepresearch-agent) 提供的多智能体拓扑思路，本项目在此基础上完成了微服务化解耦与隔离机制重构。
- **协议支持**：基于 [Model Context Protocol](https://modelcontextprotocol.io/) 实现跨服务工具调用。
- **基础设施**：感谢 [LangGraph](https://langchain-ai.github.io/langgraph)（编排）、[Docling](https://github.com/DS4SD/docling)（解析）、[Chroma](https://www.trychroma.com/)（会话存储）与 Web Search 生态社区的持续支持。

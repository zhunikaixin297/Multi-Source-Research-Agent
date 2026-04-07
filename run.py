import subprocess
import sys
import os
import time
import socket
import platform
from dotenv import load_dotenv

# ================= 配置区域 =================
FRONTEND_DIR = "src/frontend"  # 前端目录
MODEL_ID = "BAAI/bge-reranker-base" # ModelScope 模型 ID
MODELS_ROOT = "models" # 本地模型存放根目录
# ===========================================

def wait_for_port(port, host='127.0.0.1', timeout=120, service_name="Service"):
    """
    检测端口是否开启（TCP Socket 探测）
    """
    start_time = time.time()
    print(f"⏳ 等待 {service_name} (Port {port}) 就绪...", end="", flush=True)
    
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f" ✅ 就绪！")
                return True
        except (OSError, ConnectionRefusedError):
            if time.time() - start_time > timeout:
                print(f" ❌ 超时！")
                return False
            time.sleep(0.5) # 每0.5秒检测一次
            print(".", end="", flush=True)

def check_and_download_model():
    """
    检查并使用 modelscope 命令行下载模型
    """
    print(f"\n📦 [Pre-check] 检查模型环境...")
    
    # 1. 检查路径：计算完整的模型路径用于检查 (models/BAAI/bge-reranker-base)
    full_model_path = os.path.join(MODELS_ROOT, *MODEL_ID.split("/"))

    # 检查 config.json 是否存在于完整路径下
    if os.path.exists(full_model_path) and os.path.exists(os.path.join(full_model_path, "config.json")):
        print(f"✅ 模型已存在于: {full_model_path}")
        return

    # 2. 如果不存在，下载到 models 根目录
    # 注意：根据 modelscope 行为，指定 --local_dir models 会自动在其下创建 BAAI/bge-reranker-base
    print(f"⬇️  未检测到模型，正在调用命令行下载: {MODEL_ID} ...")
    print(f"    下载目标根目录: {full_model_path}")

    try:
        # 使用 subprocess 调用命令行
        # check=True 会在命令返回非零退出码时抛出 CalledProcessError
        cmd = f"modelscope download --model {MODEL_ID} --local_dir {full_model_path}"
        subprocess.run(cmd, shell=True, check=True, env=os.environ)
        
        print("✅ 模型下载命令执行完毕！")
    except subprocess.CalledProcessError:
        print("❌ 错误: 模型下载失败。")
        print("👉 请确保已安装 modelscope 命令行工具: pip install modelscope")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        sys.exit(1)

def check_and_install_frontend_deps():
    """
    检查前端 node_modules 是否存在，不存在则执行 npm install
    """
    print(f"\n📦 [Pre-check] 检查前端依赖...")
    
    # 检查 node_modules 是否存在
    node_modules_path = os.path.join(FRONTEND_DIR, "node_modules")
    
    if os.path.exists(node_modules_path) and os.path.isdir(node_modules_path):
        print(f"✅ 前端依赖 (node_modules) 已存在，跳过安装。")
        return

    print(f"⬇️  未检测到 node_modules，正在执行 npm install ...")
    print(f"    执行目录: {FRONTEND_DIR}")

    try:
        # 在前端目录下执行 npm install
        # shell=True 是必须的，以便在 Windows 上找到 npm 命令
        subprocess.run("npm install", cwd=FRONTEND_DIR, shell=True, check=True, env=os.environ)
        print("✅ 前端依赖安装完成！")
    except subprocess.CalledProcessError:
        print("❌ 错误: npm install 失败。")
        print("👉 请确保已安装 Node.js，或者尝试手动进入 src/frontend 执行 npm install")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        sys.exit(1)

def run_services():
    print("🚀 [DeepResearch Agent] 严格顺序启动脚本")
    print("--------------------------------------------------")

    # 1. 加载环境变量
    print("📂 [Init] 正在加载 .env 环境变量...")
    load_dotenv(override=True)
    os.environ["PYTHONUTF8"] = "1"

    # ========================================================
    # 阶段 0: 环境预检 (模型下载 & 前端依赖)
    # ========================================================
    check_and_download_model()
    check_and_install_frontend_deps()

    processes = []

    try:
        # ========================================================
        # 阶段 1: 启动 LiteLLM Proxy (Port 4000)
        # ========================================================
        print("\n🤖 [1/3] 正在启动 LiteLLM Proxy...")
        litellm_process = subprocess.Popen(
            ["litellm", "--config", "config.yaml"],
            shell=False,
            env=os.environ
        )
        #         # 核心修复：确保环境变量包含镜像站
        # env = os.environ.copy()
        # env["HF_ENDPOINT"] = "https://hf-mirror.com" # 解决模型下载慢
        # env["PYTHONPATH"] = os.getcwd()             # 解决模块导入问题
        
        # # 【终极修复】使用绝对路径指定配置文件，防止 cwd 偏差
        # config_path = os.path.abspath("config.yaml")
        # if not os.path.exists(config_path):
        #     raise FileNotFoundError(f"配置文件未找到: {config_path}")

        # litellm_process = subprocess.Popen(
        #     ["litellm", "--config", config_path, "--detailed_debug"], # 开启 debug 模式
        #     shell=False,
        #     env=env
        # )
        processes.append(litellm_process)

        # ⛔️ 阻塞等待：直到 LiteLLM 的 4000 端口通了，才继续
        if not wait_for_port(4000, service_name="LiteLLM"):
            raise RuntimeError("LiteLLM 启动失败，端口未响应。")

        # ========================================================
        # 阶段 2: 启动 FastAPI Backend (Port 8002)
        # ========================================================
        print("\n🔌 [2/3] 正在启动 FastAPI (Backend)...")

        backend_env = os.environ.copy()
        backend_env["PYTHONPATH"] = os.getcwd()

        uvicorn_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.backend.api.server:app", "--host", "0.0.0.0", "--port", "8002"],
            env=backend_env,
            shell=False
        )
        processes.append(uvicorn_process)

        # ⛔️ 阻塞等待：直到 Backend 的 8002 端口通了，才继续
        if not wait_for_port(8002, service_name="FastAPI"):
            raise RuntimeError("FastAPI 启动失败，端口未响应。")

        # ========================================================
        # 阶段 3: 启动 Frontend (Port 5173 - 默认 Vite 端口)
        # ========================================================
        print(f"\n💻 [3/3] 正在启动前端 (npm run dev)...")
        npm_cmd = "npm run dev -- --host 127.0.0.1 --port 5173"
        npm_process = subprocess.Popen(
            npm_cmd,
            cwd=FRONTEND_DIR,
            shell=True,
            env=os.environ
        )
        processes.append(npm_process)
        
        # 可选：也等待前端端口就绪
        if not wait_for_port(5173, service_name="Frontend"):
            raise RuntimeError("Frontend 启动失败，端口未响应。")

        print("\n--------------------------------------------------")
        print("✨ 完美！所有服务已按顺序启动完毕。")
        print("   1. LiteLLM Proxy: http://localhost:4000 (Ready)")
        print("   2. API Backend:   http://localhost:8002 (Ready)")
        print("   3. Frontend:      http://localhost:5173 (Running)")
        print("--------------------------------------------------")
        print("👉 按 Ctrl+C 可以一次性停止所有服务")

        # 挂起主进程
        uvicorn_process.wait()
        npm_process.wait()

    except KeyboardInterrupt:
        print("\n\n🛑 正在停止所有服务...")
    except RuntimeError as e:
        print(f"\n❌ 错误: {e}")
    finally:
        # 清理逻辑
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass
            
            if p.poll() is None:
                if platform.system().lower().startswith("win"):
                    subprocess.run(f"taskkill /F /T /PID {p.pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    try:
                        p.kill()
                    except Exception:
                        pass
                
        print("✅ 所有服务已清理。")

if __name__ == "__main__":
    run_services()

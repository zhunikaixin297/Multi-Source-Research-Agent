
import requests
import os
import uuid
import time
import json
import socket
import pytest

BASE_URL = "http://localhost:8002"

def test_full_flow():
    host = "localhost"
    port = 8002
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex((host, port)) != 0:
            pytest.skip("Backend service not running on localhost:8002")

    # 1. Create Workspace
    print("--- 1. Creating Workspace ---")
    resp = requests.post(f"{BASE_URL}/api/workspaces")
    resp.raise_for_status()
    workspace_data = resp.json()
    workspace_id = workspace_data["workspace_id"]
    print(f"Workspace created: {workspace_id}")

    # 2. Upload Document
    print("\n--- 2. Uploading Document ---")
    sample_doc_path = "tests/sample_documents/simple.pdf"
    if not os.path.exists(sample_doc_path):
        pytest.skip("Sample document missing: tests/sample_documents/simple.pdf")
    with open(sample_doc_path, "rb") as f:
        files = {"file": ("simple.pdf", f, "application/pdf")}
        resp = requests.post(f"{BASE_URL}/api/workspaces/{workspace_id}/documents", files=files)
    
    resp.raise_for_status()
    upload_data = resp.json()
    print(f"Document uploaded: {upload_data}")
    assert upload_data["status"] == "parsed"

    # 3. Start Task
    print("\n--- 3. Starting Task ---")
    task_payload = {
        "workspace_id": workspace_id,
        "goal": "Summarize the content of the uploaded document.",
        "max_search_iterations": 1
    }
    resp = requests.post(f"{BASE_URL}/api/tasks", json=task_payload)
    resp.raise_for_status()
    task_data = resp.json()
    task_id = task_data["task_id"]
    print(f"Task started: {task_id}")

    # 4. Check Task Streaming
    print("\n--- 4. Checking Task Streaming ---")
    goal = task_payload["goal"]
    params = {
        "workspace_id": workspace_id,
        "goal": goal
    }
    # Using stream=True to handle SSE
    with requests.get(f"{BASE_URL}/api/tasks/{task_id}/stream", params=params, stream=True) as stream_resp:
        stream_resp.raise_for_status()
        print("Connected to stream. Receiving events...")
        count = 0
        for line in stream_resp.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    event_data = json.loads(decoded_line[6:])
                    print(f"Event: {event_data.get('type')}")
                    count += 1
                    # Stop after receiving a few events to save time
                    if count >= 3:
                        break
    print("Stream check finished.")

    # 5. Delete Workspace
    print("\n--- 5. Deleting Workspace ---")
    resp = requests.delete(f"{BASE_URL}/api/workspaces/{workspace_id}")
    resp.raise_for_status()
    print(f"Workspace deleted: {resp.json()}")

if __name__ == "__main__":
    try:
        test_full_flow()
        print("\n✅ Full flow test passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

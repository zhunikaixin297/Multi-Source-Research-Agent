import asyncio
import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.backend.infrastructure.llm.factory import get_research_llm
from langchain_core.messages import HumanMessage

async def test_concurrency():
    llm = get_research_llm()
    print(f"LLM type: {type(llm)}")
    
    start_time = time.time()
    
    # Run 3 requests in parallel
    # Even if RESEARCH_LLM_MAX_CONCURRENCY=1, they should run sequentially
    tasks = [
        llm.ainvoke([HumanMessage(content="Say hi")]),
        llm.ainvoke([HumanMessage(content="Say hi")]),
        llm.ainvoke([HumanMessage(content="Say hi")])
    ]
    
    print("Sending 3 requests in parallel...")
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"Total time for 3 requests: {end_time - start_time:.2f}s")
    for i, res in enumerate(results):
        print(f"Result {i}: {res.content[:20]}...")

if __name__ == "__main__":
    asyncio.run(test_concurrency())

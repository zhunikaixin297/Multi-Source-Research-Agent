import asyncio
import os
import sys
import logging
from pprint import pprint

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.backend.infrastructure.agents.orchestrator_agent import get_orchestrator_graph
from src.backend.infrastructure.agents.states import MainState
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_full_agent_flow():
    """
    Test script to verify the full orchestrator agent flow (Planner -> Worker -> Reflector -> Reporter).
    """
    # 1. Initialize the graph
    graph = get_orchestrator_graph()
    
    # 2. Define the initial state
    # Example research goal: Researching the construction process of enterprise content knowledge base
    initial_state = {
        "goal": "研究企业内容知识库的构造流程，包括完整的文档处理和查询检索逻辑",
        "plan": [],
        "results": [],
        "user_feedback": None,
        "reflection_feedback": None,
        "loop_count": 0,
        "final_report": ""
    }
    
    # 3. Setup configuration with thread_id for persistence (in-memory sqlite usually)
    config = RunnableConfig(
        configurable={"thread_id": "test_thread_" + os.urandom(4).hex()}
    )
    
    logger.info("=" * 50)
    logger.info("Starting Orchestrator Agent Flow...")
    logger.info(f"Goal: {initial_state['goal']}")
    logger.info("=" * 50)
    
    # 4. Run the graph
    # Using astream to handle interrupts and see intermediate steps
    async for event in graph.astream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_state in event.items():
            logger.info(f"\n[Node: {node_name}] Output received.")
            
            # Print specific updates based on the node
            if node_name == "planner_agent":
                plan = node_state.get("plan", [])
                logger.info(f"--- Planner generated {len(plan)} tasks. ---")
                for i, task in enumerate(plan):
                    logger.info(f"  {i+1}. {task.title} (Query: {task.query})")
            
            elif node_name == "research_worker":
                results = node_state.get("results", [])
                logger.info(f"--- Worker completed tasks. Total results: {len(results)} ---")
                if results:
                    last_res = results[-1]
                    logger.info(f"  Latest Result: {last_res.title}")
                    # logger.info(f"  Summary: {last_res.summary[:100]}...")

            elif node_name == "reflector_agent":
                is_sufficient = node_state.get("is_sufficient")
                logger.info(f"--- Reflector Evaluation: Sufficient? {is_sufficient} ---")
                if not is_sufficient:
                    logger.info(f"  Knowledge Gap: {node_state.get('knowledge_gap')}")

            elif node_name == "reporter_agent":
                logger.info("--- Reporter generated the final report. ---")

        # Handle Interrupts (Human-in-the-loop)
        state_snapshot = await graph.aget_state(config)
        if state_snapshot.next == ("human_review",):
            logger.info("\n" + "!" * 40)
            logger.info("Graph Paused for Human Review at 'human_review' node.")
            logger.info("Auto-approving the plan for automation test...")
            logger.info("!" * 40 + "\n")
            
            # Resume by sending approval command
            async for _ in graph.astream(
                Command(resume={"action": "approve"}),
                config=config,
                stream_mode="updates"
            ):
                pass

    # 5. Retrieve final results from the graph state
    final_state = await graph.aget_state(config)
    values = final_state.values
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESEARCH REPORT")
    logger.info("=" * 60)
    print(values.get("final_report", "No report generated."))
    
    logger.info("\n" + "=" * 60)
    logger.info("EXECUTION SUMMARY")
    logger.info(f"Total Research Loops: {values.get('loop_count')}")
    logger.info(f"Total Tasks Executed: {len(values.get('results', []))}")
    logger.info("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(run_full_agent_flow())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    except Exception as e:
        logger.exception(f"Test failed with error: {e}")

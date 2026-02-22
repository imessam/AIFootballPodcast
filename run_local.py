import sys
import os
import asyncio
from dotenv import load_dotenv

# Add current directory to path
base_path = os.path.abspath(os.path.dirname(__file__))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from modules.langgraph_agent import FootballPodcastAgent

async def main():
    load_dotenv()
    
    print("--- [Main] Starting Local Football Podcast Agent (LangGraph Class-Based) ---")
    
    # Check if a query was provided, otherwise use default
    query = "Today's football highlights"
    if len(sys.argv) > 1:
        query = sys.argv[1]
    
    print(f"--- [Main] Query: {query} ---")
    
    # Initialize the Agent
    agent = FootballPodcastAgent()
    
    # Run the Agent
    final_state = await agent.run(query)
    
    print("\n--- [Main] Execution Complete ---")
    
    if final_state.get("errors"):
        print("Errors encountered:")
        for error in final_state["errors"]:
            print(f"- {error}")
    
    if final_state.get("script"):
        print("\n--- [Main] Generated Script ---")
        print(final_state["script"])
        print("-------------------------------\n")

    if final_state.get("audio_path"):
        print(f"Success! Podcast audio generated at: {final_state['audio_path']}")
    else:
        print("\nFailed to generate podcast audio.")

if __name__ == "__main__":
    asyncio.run(main())

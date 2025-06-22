import sys
import os 

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

import asyncio
import logging

logging.basicConfig(level=logging.ERROR)

from dotenv import load_dotenv

from modules.agents_podcast import PodcastAgents

load_dotenv()

# --- Verify Keys (Optional Check) ---
print("API Keys Set:")
print(f"Google API Key set: {'Yes' if os.environ.get('GOOGLE_API_KEY') and os.environ['GOOGLE_API_KEY'] != 'YOUR_GOOGLE_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")
print(f"GOOGLE_GENAI_USE_VERTEXAI Key set: {'Yes' if os.environ.get('GOOGLE_GENAI_USE_VERTEXAI') and os.environ['GOOGLE_GENAI_USE_VERTEXAI'] else 'No (REPLACE PLACEHOLDER!)'}")

# --- Define Model Constants for easier use ---

async def main(query: str):

    try:
        podcast_agent_object = PodcastAgents()

        # Create the agents.
        podcast_agent_object.create_agents()

        # Create the session service and runner.
        await podcast_agent_object.init_session()
        await podcast_agent_object.init_runner()
        
        response = await podcast_agent_object.query(query=query)

        print(f"Response from the podcast agent: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python run.py <query>")
        sys.exit(1)

    # Get the query from command line arguments
    query = sys.argv[1] 

    asyncio.run(main(query=query))
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
from modules.callbacks import *
from modules.utils import *

load_dotenv()

# --- Verify Keys (Optional Check) ---
print("API Keys Set:")
print(f"Google API Key set: {'Yes' if os.environ.get('GOOGLE_API_KEY') and os.environ['GOOGLE_API_KEY'] != 'YOUR_GOOGLE_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")
print(f"GOOGLE_GENAI_USE_VERTEXAI Key set: {'Yes' if os.environ.get('GOOGLE_GENAI_USE_VERTEXAI') and os.environ['GOOGLE_GENAI_USE_VERTEXAI'] else 'No (REPLACE PLACEHOLDER!)'}")

# --- Define Model Constants for easier use ---

async def main():

    try:
        podcast_agent_object = PodcastAgents()

        # Create the agents.
        podcast_agent_object.create_agents()

        # Create the session service and runner.
        await podcast_agent_object.init_session()
        await podcast_agent_object.init_runner()


        runner = podcast_agent_object.runner
        app_name = podcast_agent_object.app_name
        user_id = podcast_agent_object.user_id
        session_id = podcast_agent_object.session_id

        if runner is None:
            print("Error: Runner not initialized.")
            return
        
        await call_agent_async("Last Champions League Final",runner, user_id, session_id)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
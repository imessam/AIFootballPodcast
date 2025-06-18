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

MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"

async def main():

    try:
       
        app_name = "Podcast"
        user_id = "user-123"
        session_id = "session-123"

        podcast_agent_object = PodcastAgents(app_name, user_id, session_id)

        # Create the agents.
        podcast_agent_object.create_agents()

        # Create the session service and runner.
        await podcast_agent_object.init_session()
        await podcast_agent_object.init_runner()
        runner = podcast_agent_object.runner
        if runner is None:
            print("Error: Runner not initialized.")
            return
        
        await run_conversation(runner, user_id, session_id)

        # # Define initial state data - user prefers Celsius initially
        # initial_state = {
        #     "user_preference_temperature_unit": "Celsius"
        # }

    #     session_service = await create_session(app_name, user_id, session_id, initial_state)
    #     runner = create_runner(root_agent, session_service, app_name)

    #     session =  session_service.sessions[app_name][user_id][session_id]
    
        
    #     if session: session.state["user_preference_temperature_unit"] = "Fahrenheit"
        
    #     await run_conversation(runner, user_id, session_id)

    #      # --- Inspect final session state after the conversation ---
    #     # This block runs after either execution method completes.
    #     print("\n--- Inspecting Final Session State ---")
    #     final_session = await session_service.get_session(app_name=app_name,
    #                                                         user_id= user_id,
    #                                                         session_id=session_id)
    #     if final_session:
    #         # Use .get() for safer access to potentially missing keys
    #         print(f"Final Preference: {final_session.state.get('user_preference_temperature_unit', 'Not Set')}")
    #         print(f"Final Last Weather Report (from output_key): {final_session.state.get('last_weather_report', 'Not Set')}")
    #         print(f"Final Last City Checked (by tool): {final_session.state.get('last_city_checked_stateful', 'Not Set')}")
    #         # Print full state for detailed view
    #         # print(f"Full State Dict: {final_session.state}") # For detailed view
    #     else:
    #         print("\n❌ Error: Could not retrieve final session state.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
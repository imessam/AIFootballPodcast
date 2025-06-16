import sys
import os 

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

import asyncio
import logging

logging.basicConfig(level=logging.ERROR)

from google.adk.agents import Agent
from dotenv import load_dotenv

from modules.tools import *
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

    global root_agent

    try:
        app_name = "weather_tutorial_app"
        user_id = "user_1"
        session_id = "session_001"

       
        greetings_agent = Agent(
            name="greetings_agent",
            model=MODEL_GEMINI_2_0_FLASH,
            description="Handles simple greetings and hellos using the 'say_hello' tool.",
            instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting to the user. "
                    "Use the 'say_hello' tool to generate the greeting. "
                    "If the user provides their name, make sure to pass it to the tool. "
                    "Do not engage in any other conversation or tasks.",
            tools=[say_hello],
            before_model_callback=block_keyword_guardrail,
            before_tool_callback=block_paris_tool_guardrail
        )
        farwell_agent = Agent(
            name="farwell_agent",
            model=MODEL_GEMINI_2_0_FLASH,
            description="Handles simple farewells using the 'say_goodbye' tool.",
            instruction="You are the Farewell Agent. Your ONLY task is to provide a friendly farewell to the user. "
                    "Use the 'say_goodbye' tool to generate the farewell. "
                    "Do not engage in any other conversation or tasks.",
            tools=[say_goodbye],
            before_model_callback=block_keyword_guardrail,
            before_tool_callback=block_paris_tool_guardrail
        )

        root_agent = Agent(
            name="weather_agent_v2", # Give it a new version name
            model=MODEL_GEMINI_2_0_FLASH, # Use the same model=root_agent_model,
            description="The main coordinator agent. Handles weather requests and delegates greetings/farewells to specialists.",
            instruction="You are the main Weather Agent coordinating a team. Your primary responsibility is to provide weather information. "
                        "Use the 'get_weather' tool ONLY for specific weather requests (e.g., 'weather in London'). "
                        "You have specialized sub-agents: "
                        "1. 'greeting_agent': Handles simple greetings like 'Hi', 'Hello'. Delegate to it for these. "
                        "2. 'farewell_agent': Handles simple farewells like 'Bye', 'See you'. Delegate to it for these. "
                        "Analyze the user's query. If it's a greeting, delegate to 'greeting_agent'. If it's a farewell, delegate to 'farewell_agent'. "
                        "If it's a weather request, handle it yourself using 'get_weather'. "
                        "For anything else, respond appropriately or state you cannot handle it.",
            tools=[get_weather_stateful], # Root agent still needs the weather tool for its core task
            before_model_callback=block_keyword_guardrail,
            before_tool_callback=block_paris_tool_guardrail,
            # Key change: Link the sub-agents here!
            sub_agents=[greetings_agent, farwell_agent],
            output_key="last_weather_report"
        )

        # Define initial state data - user prefers Celsius initially
        initial_state = {
            "user_preference_temperature_unit": "Celsius"
        }

        session_service = await create_session(app_name, user_id, session_id, initial_state)
        runner = create_runner(root_agent, session_service, app_name)

        session =  session_service.sessions[app_name][user_id][session_id]
    
        
        if session: session.state["user_preference_temperature_unit"] = "Fahrenheit"
        
        await run_conversation(runner, user_id, session_id)

         # --- Inspect final session state after the conversation ---
        # This block runs after either execution method completes.
        print("\n--- Inspecting Final Session State ---")
        final_session = await session_service.get_session(app_name=app_name,
                                                            user_id= user_id,
                                                            session_id=session_id)
        if final_session:
            # Use .get() for safer access to potentially missing keys
            print(f"Final Preference: {final_session.state.get('user_preference_temperature_unit', 'Not Set')}")
            print(f"Final Last Weather Report (from output_key): {final_session.state.get('last_weather_report', 'Not Set')}")
            print(f"Final Last City Checked (by tool): {final_session.state.get('last_city_checked_stateful', 'Not Set')}")
            # Print full state for detailed view
            # print(f"Full State Dict: {final_session.state}") # For detailed view
        else:
            print("\n❌ Error: Could not retrieve final session state.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
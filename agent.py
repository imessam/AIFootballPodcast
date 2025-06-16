# @title Import necessary libraries
import os
import asyncio

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts

from tools import *
from callbacks import *

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

print("Libraries imported.")


# @title Configure API Keys (Replace with your actual keys!)

# --- IMPORTANT: Replace placeholders with your real API keys ---

# Gemini API Key (Get from Google AI Studio: https://aistudio.google.com/app/apikey)
os.environ["GOOGLE_API_KEY"] = "AIzaSyB0VXsKy_DsFIL7GrwbUwkdQM2WVCjNF9A" # <--- REPLACE

# [Optional]
# OpenAI API Key (Get from OpenAI Platform: https://platform.openai.com/api-keys)
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY' # <--- REPLACE

# [Optional]
# Anthropic API Key (Get from Anthropic Console: https://console.anthropic.com/settings/keys)
os.environ['ANTHROPIC_API_KEY'] = 'YOUR_ANTHROPIC_API_KEY' # <--- REPLACE

# --- Verify Keys (Optional Check) ---
print("API Keys Set:")
print(f"Google API Key set: {'Yes' if os.environ.get('GOOGLE_API_KEY') and os.environ['GOOGLE_API_KEY'] != 'YOUR_GOOGLE_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")
print(f"OpenAI API Key set: {'Yes' if os.environ.get('OPENAI_API_KEY') and os.environ['OPENAI_API_KEY'] != 'YOUR_OPENAI_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")
print(f"Anthropic API Key set: {'Yes' if os.environ.get('ANTHROPIC_API_KEY') and os.environ['ANTHROPIC_API_KEY'] != 'YOUR_ANTHROPIC_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")

# Configure ADK to use API keys directly (not Vertex AI for this multi-model setup)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"


# --- Define Model Constants for easier use ---

# More supported models can be referenced here: https://ai.google.dev/gemini-api/docs/models#model-variations
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"

print("\nEnvironment configured.")


def create_agent(
        name = "weather_agent_v1",
        AGENT_MODEL = MODEL_GEMINI_2_0_FLASH, 
        description = "Provides weather information for specific cities.", 
        instruction = "You are a helpful weather assistant.", 
        tools = [get_weather],
        sub_agents = [],
        output_key = "final_response") -> Agent:

    # @title Define the Weather Agent
    # Use one of the model constants defined earlier

    weather_agent = Agent(
        name=name,
        model=AGENT_MODEL, # Can be a string for Gemini or a LiteLlm object
        description=description,
        instruction=instruction,
        tools=tools,
        sub_agents=sub_agents, # Pass the function directly
        output_key = output_key,
        before_model_callback=block_keyword_guardrail,
        before_tool_callback=block_paris_tool_guardrail
    )

    print(f"Agent '{weather_agent.name}' created using model '{AGENT_MODEL}'.")

    return weather_agent



async def create_session(app_name : str, 
                        user_id : str, 
                        session_id : str,
                        initial_state = {}) -> InMemorySessionService:
  

    # @title Setup Session Service and Runner

    # --- Session Management ---
    # Key Concept: SessionService stores conversation history & state.
    # InMemorySessionService is simple, non-persistent storage for this tutorial.
    session_service = InMemorySessionService()

    # Create the specific session where the conversation will happen
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=initial_state
    )
    print(f"Session created: App='{app_name}', User='{user_id}', Session='{session_id}'")

    
    # Verify the initial state was set correctly
    retrieved_session = await session_service.get_session(app_name=app_name,
                                                            user_id=user_id,
                                                            session_id = session_id)

    print("\n--- Initial Session State ---")
    if retrieved_session:
        print(retrieved_session.state)
    else:
        print("Error: Could not retrieve session.")

    return session_service


def create_runner(agent : Agent, 
                  session_service : InMemorySessionService, 
                  app_name : str) -> Runner:

    # --- Runner ---
    # Key Concept: Runner orchestrates the agent execution loop.
    runner = Runner(
        agent=agent, # The agent we want to run
        app_name=app_name,   # Associates runs with our app
        session_service=session_service # Uses our session manager
    )
    print(f"Runner created for agent '{runner.agent.name}'.")

    return runner

# @title Define Agent Interaction Function
async def call_agent_async(query: str, runner : Runner, user_id, session_id) -> str | None:
  """Sends a query to the agent and prints the final response."""
  print(f"\n>>> User Query: {query}")

  # Prepare the user's message in ADK format
  content = types.Content(role='user', parts=[types.Part(text=query)])

  final_response_text = "Agent did not produce a final response." # Default

  # Key Concept: run_async executes the agent logic and yields Events.
  # We iterate through events to find the final answer.
  async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
      # You can uncomment the line below to see *all* events during execution
    #   print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

      # Key Concept: is_final_response() marks the concluding message for the turn.
      if event.is_final_response():
          if event.content and event.content.parts:
             # Assuming text response in the first part
             final_response_text = event.content.parts[0].text
          elif event.actions and event.actions.escalate: # Handle potential errors/escalations
             final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
          # Add more checks here if needed (e.g., specific error codes)
          break # Stop processing events once the final response is found

  print(f"<<< Agent Response: {final_response_text}")

  return final_response_text


# @title Run the Initial Conversation

# We need an async function to await our interaction helper
async def run_conversation(runner : Runner, user_id : str, session_id : str):

    await call_agent_async("Hi there! I'm Mohamed",
                                       runner=runner,
                                       user_id=user_id,
                                       session_id=session_id)
    
    await call_agent_async("What is the weather like in London?",
                                       runner=runner,
                                       user_id=user_id,
                                       session_id=session_id)

    await call_agent_async("How about Paris?",
                                       runner=runner,
                                       user_id=user_id,
                                       session_id=session_id) # Expecting the tool's error message

    await call_agent_async("Tell me the weather in New York",
                                       runner=runner,
                                       user_id=user_id,
                                       session_id=session_id)
    
    await call_agent_async("Farewell!",
                                       runner=runner,
                                       user_id=user_id,
                                       session_id=session_id)

async def main():
    try:
        app_name = "weather_tutorial_app"
        user_id = "user_1"
        session_id = "session_001"

       
        greetings_agent = create_agent(
            name="greetings_agent",
            AGENT_MODEL=MODEL_GEMINI_2_0_FLASH,
            description="Handles simple greetings and hellos using the 'say_hello' tool.",
            instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting to the user. "
                    "Use the 'say_hello' tool to generate the greeting. "
                    "If the user provides their name, make sure to pass it to the tool. "
                    "Do not engage in any other conversation or tasks.",
            tools=[say_hello]
        )
        farwell_agent = create_agent(
            name="farwell_agent",
            AGENT_MODEL=MODEL_GEMINI_2_0_FLASH,
            description="Handles simple farewells using the 'say_goodbye' tool.",
            instruction="You are the Farewell Agent. Your ONLY task is to provide a friendly farewell to the user. "
                    "Use the 'say_goodbye' tool to generate the farewell. "
                    "Do not engage in any other conversation or tasks.",
            tools=[say_goodbye]
        )

        weather_agent_team = create_agent(
            name="weather_agent_v2", # Give it a new version name
            AGENT_MODEL=MODEL_GEMINI_2_0_FLASH, # Use the same model=root_agent_model,
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
            # Key change: Link the sub-agents here!
            sub_agents=[greetings_agent, farwell_agent],
            output_key="last_weather_report"
        )

        # Define initial state data - user prefers Celsius initially
        initial_state = {
            "user_preference_temperature_unit": "Celsius"
        }

        session_service = await create_session(app_name, user_id, session_id, initial_state)
        runner = create_runner(weather_agent_team, session_service, app_name)

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
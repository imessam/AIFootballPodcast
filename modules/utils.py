import logging
import wave

logging.basicConfig(level=logging.ERROR)

from google.adk.agents import BaseAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts



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


def create_runner(agent : BaseAgent, 
                  session_service : InMemorySessionService, 
                  app_name : str) -> Runner:

    # --- Runner ---
    # Key Concept: Runner orchestrates the agent execution loop.
    """
    Creates a Runner for the given agent and session service.

    Args:
        agent: The Agent to run.
        session_service: The InMemorySessionService to store and retrieve
            conversation history and state.
        app_name: The name of the application that the agent is part of.

    Returns:
        A Runner object that can be used to start the agent execution loop.
    """
    runner = Runner(
        agent=agent, # The agent we want to run
        app_name=app_name,   # Associates runs with our app
        session_service=session_service # Uses our session manager
    )
    print(f"Runner created for agent '{runner.agent.name}'.")

    return runner

# @title Define Agent Interaction Function
async def call_agent_async(query: str, runner : Runner, user_id, session_id) -> str | None:
    """
    Calls the agent with the given query, user_id, and session_id asynchronously.

    Args:
        query (str): The user's query to the agent.
        runner (Runner): The Runner instance to use for the agent.
        user_id (str): The unique identifier for the user.
        session_id (str): The unique identifier for the session.

    Returns:
        str | None: The agent's final response to the query, or None if no final response is produced.
    """
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response." # Default

    agents_final_responses = {}

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # You can uncomment the line below to see *all* events during execution
        print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate: # Handle potential errors/escalations
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            # Add more checks here if needed (e.g., specific error codes)
            agents_final_responses[event.author] = final_response_text # Stop processing events once the final response is found

    print(f"<<< Agent Response: {final_response_text}")

    return final_response_text

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    
    """
    Saves PCM audio data to a WAV file.

    Args:
        filename (str): The name of the file to which the audio data will be saved.
        pcm (bytes): The PCM audio data to be saved.
        channels (int, optional): The number of audio channels. Defaults to 1.
        rate (int, optional): The sample rate (samples per second). Defaults to 24000.
        sample_width (int, optional): The sample width in bytes. Defaults to 2.
    """

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
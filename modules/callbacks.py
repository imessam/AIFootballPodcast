from google.adk.agents.callback_context import CallbackContext
from typing import Optional
from google.genai import types 
from google.adk.models.llm_response import LlmResponse

from langchain_core.utils.json import parse_json_markdown




# --- 1. Define the Callback Function ---
def check_empty_agents_state(callback_context: CallbackContext) -> Optional[types.Content]:
    """
    Callback function to check if matches were found in the session state.
    """
    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id
    current_state = callback_context.state.to_dict()

    state_to_check = "{}"

    print(f"\n[Callback] Entering agent: {agent_name} (Inv: {invocation_id})")
    print(f"[Callback] Current State: {current_state}")

    if agent_name == "web_search_agent" and "combined_matches" in current_state:

        print(f"[Callback] Agent {agent_name} has 'combined_matches' in state.")

        state_to_check = current_state.get("combined_matches", "{}")

    elif agent_name == "podcast_writer_agent" and "web_search_results" in current_state:
        print(f"[Callback] Agent {agent_name} has 'web_search_results' in state.")

        state_to_check = current_state.get("web_search_results", "{}")

    elif agent_name == "text_to_speech_agent" and "podcast_scripts" in current_state:
        print(f"[Callback] Agent {agent_name} has 'podcast_scripts' in state.")

        state_to_check = current_state.get("podcast_scripts", "{}")

    elif (agent_name == "file_uploader" or agent_name == "check_agent_called_tool") and "podcast_audio" in current_state:
        print(f"[Callback] Agent {agent_name} has 'podcast_audio' in state.")

        state_to_check = current_state.get("podcast_audio", "{}")

    else:
        print(f"[Callback] Agent {agent_name} does not have expected state keys.")

    
    state_to_check_json = parse_json_markdown(state_to_check)


    print(f"[Callback] Check State JSON: {state_to_check_json}, type: {type(state_to_check_json)}, length: {len(state_to_check_json)}")

    # Check the condition in session state dictionary
    if len(state_to_check_json.keys()) == 0:
        print(f"[Callback] State condition 'empty_state' met: Skipping agent {agent_name}.")
        # Return Content to skip the agent's run


        custom_response = f"""
            ```json
            {{
                "error": "Agent {agent_name} skipped by before_agent_callback due to state."
            }}
            ```
        """

        return types.Content(
            parts=[types.Part(text=custom_response)],
            role="model" # Assign model role to the overriding response
        )
    else:
        print(f"[Callback] State condition not met: Proceeding with agent {agent_name}.")
        # Return None to allow the LlmAgent's normal execution
        return None
    
import sys
import os 

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

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

class PodcastAgents:
    def __init__(self, 
                 coordinator_model : str = MODEL_GEMINI_2_0_FLASH,
                 web_search_model : str = MODEL_GEMINI_2_0_FLASH,
                 podcast_writer_model : str = MODEL_GEMINI_2_0_FLASH,
                 text_to_speech_model : str = MODEL_GEMINI_2_0_FLASH):
        
        self.coordinator_model = coordinator_model
        self.web_search_model = web_search_model
        self.podcast_writer_model = podcast_writer_model
        self.text_to_speech_model = text_to_speech_model

        self.coordinator_agent = None
        self.web_search_agent = None
        self.podcast_writer_agent = None
        self.text_to_speech_agent = None

        self.app_name = "football_podcast_app"
        self.user_id = "user_1"
        self.session_id = "session_001"

        self.session_service = None
        self.runner : Runner

    def _create_coordinator_agent(self, custom_instruction : str) -> bool:

        name = "coordinator_agent"
        description = "Handles the coordination of the podcast agents."
        tools = []

        default_instruction = """
                                You are the Coordinator Agent.
                                Your ONLY task is to coordinate the podcast agents.
                                Do not engage in any other conversation or tasks.
                            """

        instruction = custom_instruction if len(custom_instruction) > 0 else default_instruction

        print(f"Creating {name} with description: {description}, instruction: {instruction}, and tools: {tools}")

        try:
            self.coordinator_agent = Agent(
                name = name,
                model = self.coordinator_model,
                description = description,
                instruction = instruction,
                tools = tools,
                before_model_callback = None,
                before_tool_callback = None
            )
        except Exception as e:
            print(f"Error creating {name}: {e}")
            return False
        
        print(f"{name} created successfully.")

        return True
    
    def _create_web_search_agent(self, custom_instruction : str) -> bool:

        name = "web_search_agent"
        description = "Performs web searches."
        tools = []
        output_key = "web_search_results"

        default_instruction = """
                                You are the Web Searcher Agent.
                                Your ONLY task is to search the web.
                                Use the web search tool only to search the web.
                                Do not engage in any other conversation or tasks.
                            """

        instruction = custom_instruction if len(custom_instruction) > 0 else default_instruction

        print(f"Creating {name} with description: {description}, instruction: {instruction}, and tools: {tools}")

        try:
            self.web_search_agent = Agent(
                name = name,
                model = self.web_search_model,
                description = description,
                instruction = instruction,
                tools = tools,
                before_model_callback = None,
                before_tool_callback = None,
                output_key = output_key
            )
        except Exception as e:
            print(f"Error creating {name}: {e}")
            return False
        
        print(f"{name} created successfully.")

        return True
    
    def _create_podcast_writer_agent(self, custom_instruction : str) -> bool:

        name = "podcast_writer_agent"
        description = "Writes podcasts."
        tools = []
        output_key = "podcast_transcript"

        default_instruction = """
                                You are the Podcast Writer Agent.
                                Your ONLY task is to write podcasts.
                                Do not engage in any other conversation or tasks.
                            """

        instruction = custom_instruction if len(custom_instruction) > 0 else default_instruction

        print(f"Creating {name} with description: {description}, instruction: {instruction}, and tools: {tools}")

        try:
            self.podcast_writer_agent = Agent(
                name = name,
                model = self.podcast_writer_model,
                description = description,
                instruction = instruction,
                tools = tools,
                before_model_callback = None,
                before_tool_callback = None,
                output_key = output_key,
            )
        except Exception as e:
            print(f"Error creating {name}: {e}")
            return False
        
        print(f"{name} created successfully.")

        return True
    
    
    def _create_text_to_speech_agent(self, custom_instruction : str) -> bool:

        name = "text_to_speech_agent"
        description = "Converts text to speech."
        tools = []
        output_key = "podcast_audio"

        default_instruction = """
                                You are the Text to Speech Agent.
                                Your ONLY task is to convert text to speech.
                                Do not engage in any other conversation or tasks.
                            """

        instruction = custom_instruction if len(custom_instruction) > 0 else default_instruction

        print(f"Creating {name} with description: {description}, instruction: {instruction}, and tools: {tools}")

        try:
            self.text_to_speech_agent = Agent(
                name = name,
                model = self.text_to_speech_model,
                description = description,
                instruction = instruction,
                tools = tools,
                before_model_callback = None,
                before_tool_callback = None,
                output_key = output_key
            )
        except Exception as e:
            print(f"Error creating {name}: {e}")
            return False
        
        print(f"{name} created successfully.")

        return True
    
    def create_agents(self, custom_instructions : dict = {}):

        print(f"Creating agents with custom instructions: {custom_instructions}")

        coordinator_instruction = custom_instructions.get("coordinator", "")
        web_search_instruction = custom_instructions.get("web_search", "")
        podcast_writer_instruction = custom_instructions.get("podcast_writer", "")
        text_to_speech_instruction = custom_instructions.get("text_to_speech", "")

        self._create_coordinator_agent(coordinator_instruction)
        self._create_web_search_agent(web_search_instruction)
        self._create_podcast_writer_agent(podcast_writer_instruction)
        self._create_text_to_speech_agent(text_to_speech_instruction)

        print("Agents created successfully.")

        return

    async def init_session(self):

        print(f"Initializing session: {self.app_name}, {self.user_id}, {self.session_id}")

        self.session_service = await create_session(self.app_name,
                                                    self.user_id, 
                                                    self.session_id)

        print(f"Session initialized: {self.app_name}, {self.user_id}, {self.session_id}")

        return
    
    async def init_runner(self):

        print(f"Initializing runner: {self.app_name}, {self.user_id}, {self.session_id}")

        if not self.session_service:
            print(f"Error: Session service not initialized ... ")
            return
        
        if not self.coordinator_agent:
            print(f"Error: Coordinator agent not initialized ... ")
            return

        self.runner = create_runner(self.coordinator_agent, self.session_service, self.app_name)

        print(f"Runner initialized: {self.app_name}, {self.user_id}, {self.session_id}")

        return
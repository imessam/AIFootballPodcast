import sys
import os 

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

import logging

logging.basicConfig(level=logging.ERROR)

from google.adk.agents import Agent, SequentialAgent

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
                 matches_fetcher_model : str = MODEL_GEMINI_2_0_FLASH,
                 web_search_model : str = MODEL_GEMINI_2_0_FLASH,
                 podcast_writer_model : str = MODEL_GEMINI_2_0_FLASH,
                 text_to_speech_model : str = MODEL_GEMINI_2_0_FLASH):
        
        self.matches_fetcher_model = matches_fetcher_model
        self.web_search_model = web_search_model
        self.podcast_writer_model = podcast_writer_model
        self.text_to_speech_model = text_to_speech_model

        self.matches_fetcher_agent = None
        self.web_search_agent = None
        self.podcast_writer_agent = None
        self.text_to_speech_agent = None

        self.sequential_agent = None

        self.app_name = "football_podcast_app"
        self.user_id = "user_1"
        self.session_id = "session_001"

        self.session_service = None
        self.runner : Runner

    def _create_matches_fetcher_agent(self, custom_instruction : str) -> bool:

        name = "matches_fetcher"
        description = "Fetches matches from the given tool."
        tools = [get_matches_by_date]
        output_key = "matches"

        default_instruction = """
                                You are the Match Fetcher Agent.
                                Your ONLY task is to fetch matches from the given date using the `get_matches_by_date` tool.
                                The date is provided in the format YYYY-MM-DD.
                                You will receive the response as a dictionary with a status key set to "success" in case matches are found, "error" in case no matches are found.
                                In case of matches are found, the result will contain a dictionary with match ID as the key,
                                and a dictionary containing the competition, home team, away team, home score and away score as the value if matches are found,
                                or an error message if no matches are found, eg:

                                {
                                    "status": "success",
                                    "matches": {
                                        "match_id_1": {
                                            "competition": "Premier League",
                                            "home_team": "Team A",
                                            "away_team": "Team B",
                                            "home_score": 2,
                                            "away_score": 1
                                        },
                                        "match_id_2": {
                                            "competition": "La Liga",
                                            "home_team": "Team C",
                                            "away_team": "Team D",
                                            "home_score": 3,
                                            "away_score": 0
                                        }
                                    }
                                }

                                or an error message if no matches are found, eg:
                                {
                                    "status": "error",
                                    "error": "No matches found for the given date."
                                }

                                Group the matches in the same copetition. Return the matches in the format:
                                {
                                    "Premier League":
                                    {
                                        "match_id_1": {
                                            "home_team": "Team A",
                                            "away_team": "Team B",
                                            "home_score": 2,
                                            "away_score": 1
                                        }
                                    },
                                    "La Liga":
                                    {
                                        "match_id_1": {
                                            "home_team": "Team C",
                                            "away_team": "Team D",
                                            "home_score": 3,
                                            "away_score": 0
                                        }
                                    }
                                }
                                If no matches are found, return an empty JSON object: {}.
                                You will receive the date to fetch matches for in the format YYYY-MM-DD.
                                Do not engage in any other conversation or tasks.
                            """

        instruction = custom_instruction if len(custom_instruction) > 0 else default_instruction

        print(f"Creating {name} with description: {description}, instruction: {instruction}, and tools: {tools}")

        try:
            self.matches_fetcher_agent = Agent(
                name = name,
                model = self.matches_fetcher_model,
                description = description,
                instruction = instruction,
                tools = [get_matches_by_date],
                before_model_callback = None,
                before_tool_callback = None,
                output_key = output_key
            )
        except Exception as e:
            print(f"Error creating {name}: {e}")
            return False
        
        print(f"{name} created successfully.")

        return True
    
    def _create_web_search_agent(self, custom_instruction : str) -> bool:

        name = "web_search_agent"
        description = "Performs web searches."
        tools = [google_search]
        output_key = "web_search_results"

        default_instruction = """
                                You are the Web Searcher Agent.
                                Your ONLY task is to search the web using the `google_search` tool.
                                Search the web for information about the matches provided by the Matches Fetcher Agent.
                                The matches are provided in the format: 
                                {
                                    "competition_name_1":
                                        {
                                            "match_id_1": {
                                                "home_team": "Team A",
                                                "away_team": "Team B",
                                                "home_score": home_score,
                                                "away_score": away_score
                                            },
                                        }
                                    "competition_name_2":
                                        {
                                            "match_id_1": {
                                                "home_team": "Team C",
                                                "away_team": "Team D",
                                                "home_score": home_score,
                                                "away_score": away_score
                                            },
                                        }
                                }
                                If you receive an empty JSON object, that means no matches were found, so you should not search the web, you should return an empty JSON object.
                                Search for the latest news, statistics, and any other relevant information about the matches.
                                Then, generate a summary of the search results.
                                Output the results in a JSON format like this: 
                                {
                                    "competition_name_1":
                                        {
                                            "match_id_1": {
                                                "summary": "summary of the search results",
                                                "details": "detailed information about the match"
                                            },
                                        }
                                    "competition_name_2":
                                        {
                                            "match_id_1": {
                                                "summary": "summary of the search results",
                                                "details": "detailed information about the match"
                                            },
                                        }
                                }
                                If you receive an empty JSON object, return an empty JSON object: {}.
                                If no information is found, return an empty JSON object: {}.
                                Do not engage in any other conversation or tasks.

                                Here are the matches you need to search for:
                                {matches}
                            """

        instruction = custom_instruction if len(custom_instruction) > 0 else default_instruction

        print(f"Creating {name} with description: {description}, instruction: {instruction}, and tools: {tools}")

        try:
            self.web_search_agent = Agent(
                name = name,
                model = self.web_search_model,
                description = description,
                instruction = instruction,
                tools = [google_search],
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
        output_key = "podcast_scripts"

        default_instruction = """
                                You are the Podcast Writer Agent.
                                Your ONLY task is to write podcasts.
                                You will receive web search results from the Web Search Agent for the matches provided by the Matches Fetcher Agent.
                                Use the web search results to write a podcast script.
                                The web search results are provided in the format:
                                {
                                    "competition_name_1":
                                        {
                                            "match_id_1": {
                                                "summary": "summary of the search results",
                                                "details": "detailed information about the match"
                                            },
                                        }
                                    "competition_name_2":
                                        {
                                            "match_id_1": {
                                                "summary": "summary of the search results",
                                                "details": "detailed information about the match"
                                            },
                                        }
                                }
                                If you receive an empty JSON object, that means no matches were found, so you should not write a podcast script, you should return an empty JSON object.
                                Write a podcast script for only one match, choose the one with the most relevant information.
                                Each podcast script should be a detailed and engaging narrative about the match, including key moments, player performances, and any other relevant information.
                                The podcast is about a football match, so it should be written in a conversational tone, as if two sports commentators are discussing the match.
                                There is two speakers in the podcast, call them "Ahmed" and "Fatima".
                                The speakers should alternate in the podcast, with each speaker providing their own perspective on the match.
                                The speaker "Ahmed" should provide the main commentary, while the speaker "Fatima" should provide analysis and insights.
                                The speaker "Fatima" has a joyful and enthusiastic tone, while the speaker "Ahmed" has a more serious and analytical tone.
                                The podcast should be between 2 and 4 minutes long, so it should be concise and to the point.
                                Output the podcast script in a JSON format like this:
                                {
                                    "match_id_1": "podcast_script"
                                }
                                
                                The "podcast_script" should be another JSON object with the keys "speaker_1", "speaker_2", and "content".
                                The "speaker_1" and "speaker_2" should be the names of the speakers, "Ahmed" and "Fatima".
                                The "content" should be the actual podcast script, which should be a string containing the dialogue between the two speakers, eg:
                                {
                                    "speaker_1": "Ahmed",
                                    "speaker_2": "Fatima",
                                    "content": "Ahmed: "ahmed_first_transcript"\nFatima: "fatima_first_transcript"\nAhmed: "ahmed_second_transcript"\nFatima: "fatima_second_transcript" etc."
                                } 
                                If no information is found, return an empty JSON object: {}.
                                Do not engage in any other conversation or tasks.

                                Here are the matches web search results you need to write podcasts for:
                                {web_search_results}
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
        tools = [podcast_text_to_speech]
        output_key = "podcast_audio"

        default_instruction = """
                                You are the Text to Speech Agent.
                                Your ONLY task is to convert podcast text to speech using the `podcast_text_to_speech` tool.
                                The podcast scripts are provided in the format:
                                {
                                    "match_id": "podcast_script"
                                }
                                Pass the "podcast_script" to the `podcast_text_to_speech` tool to convert it to speech.
                                The `podcast_text_to_speech` tool will return a string containing the path to the audio file.
                            
                                Do not engage in any other conversation or tasks.
                                Here are the podcast scripts you need to convert to speech:
                                {podcast_scripts}
                            """

        instruction = custom_instruction if len(custom_instruction) > 0 else default_instruction

        print(f"Creating {name} with description: {description}, instruction: {instruction}, and tools: {tools}")

        try:
            self.text_to_speech_agent = Agent(
                name = name,
                model = self.text_to_speech_model,
                description = description,
                instruction = instruction,
                tools = [podcast_text_to_speech],
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

        matches_fetcher_instruction = custom_instructions.get("matches_fetcher", "")
        web_search_instruction = custom_instructions.get("web_search", "")
        podcast_writer_instruction = custom_instructions.get("podcast_writer", "")
        text_to_speech_instruction = custom_instructions.get("text_to_speech", "")

        self._create_matches_fetcher_agent(matches_fetcher_instruction)
        self._create_web_search_agent(web_search_instruction)
        self._create_podcast_writer_agent(podcast_writer_instruction)
        self._create_text_to_speech_agent(text_to_speech_instruction)

        if not self.matches_fetcher_agent:
            print(f"Error: Matches fetcher agent not created ... ")
            return

        if not self.web_search_agent:
            print(f"Error: Web search agent not created ... ")
            return

        if not self.podcast_writer_agent:
            print(f"Error: Podcast writer agent not created ... ")
            return

        if not self.text_to_speech_agent:
            print(f"Error: Text to speech agent not created ... ")
            return

        self.sequential_agent = SequentialAgent(
            name = "podcast_generation_pipeline",
            description = "Executes the podcast generation pipeline sequentially.",
            sub_agents = [
                            self.matches_fetcher_agent, 
                            self.web_search_agent, 
                            self.podcast_writer_agent, 
                            self.text_to_speech_agent
                          ],
        )

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
        
        if not self.sequential_agent:
            print(f"Error: Sequential agent not initialized ... ")
            return

        self.runner = create_runner(self.sequential_agent, self.session_service, self.app_name)

        print(f"Runner initialized: {self.app_name}, {self.user_id}, {self.session_id}")

        return
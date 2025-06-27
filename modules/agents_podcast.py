import sys
import os 

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

import logging

logging.basicConfig(level=logging.ERROR)

from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.tools import google_search  # Import the tool

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
MODEL_GEMINI_2_0_PRO = "gemini-2.5-pro"

class PodcastAgents:
    def __init__(self, 
                    matches_fetcher_model : str = MODEL_GEMINI_2_0_FLASH,
                    matches_web_fetcher_model : str = MODEL_GEMINI_2_0_FLASH,
                    web_search_model : str = MODEL_GEMINI_2_0_FLASH,
                    podcast_writer_model : str = MODEL_GEMINI_2_0_PRO,
                    text_to_speech_model : str = MODEL_GEMINI_2_0_PRO,
                    files_uploader_model : str = MODEL_GEMINI_2_0_FLASH) -> None:
        
        self.matches_fetcher_model = matches_fetcher_model
        self.matches_web_fetcher_model = matches_web_fetcher_model
        self.matches_combiner_model = matches_fetcher_model
        self.web_search_model = web_search_model
        self.podcast_writer_model = podcast_writer_model
        self.text_to_speech_model = text_to_speech_model
        self.files_uploader_model = files_uploader_model

        self.matches_fetcher_agent = None
        self.matches_web_fetcher_agent = None
        self.matches_parallel_agents = None
        self.matches_combiner_agent = None
        self.web_search_agent = None
        self.podcast_writer_agent = None
        self.text_to_speech_agent = None
        self.check_agent_called_tool = None
        self.text_to_speech_agent_loop = None
        self.files_uploader_agent = None
        self.sequential_agent = None

        self.app_name = "football_podcast_app"
        self.user_id = "user_1"
        self.session_id = "session_001"

        self.session_service = None
        self.runner : Runner

    def _create_matches_fetcher_agent(self, custom_instruction : str) -> bool:

        """
        Creates the Matches Fetcher Agent.

        Args:
            custom_instruction (str): A custom instruction for the agent.

        Returns:
            bool: True if the agent is created successfully, False otherwise.
        """
        
        name = "matches_fetcher"
        description = "Fetches matches from the given tool."
        tools = [get_matches_by_date]
        output_key = "matches_fetcher_results"

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
    
    def _create_matches_web_fetcher_agent(self, custom_instruction : str) -> bool:

        """
        Creates the Matches Web Fetcher Agent.

        Args:
            custom_instruction (str): A custom instruction for the agent.

        Returns:
            bool: True if the agent is created successfully, False otherwise.
        """
        
        name = "matches_web_fetcher"
        description = "Fetches matches from web."
        tools = [google_search]
        output_key = "matches_fetcher_web_results"

        default_instruction = """
                                You are the Match Web Fetcher Agent.
                                Your ONLY task is to fetch matches from the web for the given date using the `google_search` tool.
                                The date is provided in the format YYYY-MM-DD.
                                Search only for popular football competitions and matches.
                                Do not search for friendly competitions or matches.
                                Do not return more than 5 matches.
                                In case of matches are found, group the matches in the same competition. Return the matches in this exact format:
                                {
                                    "Premier League":
                                    {
                                        "match_id_1": {
                                            "home_team": "Team A",
                                            "away_team": "Team B",
                                            "home_score": home_score,
                                            "away_score": away_score
                                        }
                                    },
                                    "La Liga":
                                    {
                                        "match_id_1": {
                                            "home_team": "Team C",
                                            "away_team": "Team D",
                                            "home_score": home_score,
                                            "away_score": away_score
                                        }
                                    }
                                }
                                Replace any "null" values with 0.
                                Do not return all the matches, only return the best and most important matches.
                                If no matches are found, return an empty JSON object: {}.
                                You will receive the date to fetch matches for in the format YYYY-MM-DD.
                                Do not engage in any other conversation or tasks.
                            """

        instruction = custom_instruction if len(custom_instruction) > 0 else default_instruction

        print(f"Creating {name} with description: {description}, instruction: {instruction}, and tools: {tools}")

        try:
            self.matches_web_fetcher_agent = Agent(
                name = name,
                model = self.matches_web_fetcher_model,
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
    
    def _create_matches_combiner_agent(self, custom_instruction : str) -> bool:

        """
        Creates the Matches Combiner Agent.

        Args:
            custom_instruction (str): A custom instruction for the agent.

        Returns:
            bool: True if the agent is created successfully, False otherwise.

        """
        
        name = "matches_combiner_agent"
        description = "Combines matches."
        tools = []
        output_key = "combined_matches"

        default_instruction = """
                                You are the Matches Combiner Agent.
                                Your ONLY task is to combine matches from the Matches Fetcher Agent and the Matches Web Fetcher Agent.
                                You will receive matches from the Matches Fetcher Agent and matches from the Matches Web Fetcher Agent in the format:
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

                                Combine the matches from both sources into a single dictionary.
                                Do not duplicate any matches.
                                If you found the same match in both sources, take the match from the source that returned it first.
                                If one source returns matches and the other returns an empty JSON object, return the matches from the source that returns matches.
                                If you receive an empty JSON object from both sources, that means no matches were found, so you should not combine anything, you should return an empty JSON object.
                                Output the combined matches in the same format as above.
                                If no matches are found, return an empty JSON object: {}.
                                Do not engage in any other conversation or tasks.
                                Here are the matches you need to combine:
                                Matches from Matches Fetcher Agent:
                                {matches_fetcher_results}
                                Matches from Matches Web Fetcher Agent:
                                {matches_fetcher_web_results}
                            """
        instruction = custom_instruction if len(custom_instruction) > 0 else default_instruction

        print(f"Creating {name} with description: {description}, instruction: {instruction}, and tools: {tools}")

        try:
            self.matches_combiner_agent = Agent(
                name = name,
                model = self.matches_fetcher_model,
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

    
    def _create_web_search_agent(self, custom_instruction : str) -> bool:

        """
        Creates the Web Search Agent.

        Args:
            custom_instruction (str): A custom instruction for the agent.

        Returns:
            bool: True if the agent is created successfully, False otherwise.

        """

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
                                                "details": "detailed information about the match"
                                            },
                                        }
                                    "competition_name_2":
                                        {
                                            "match_id_1": {
                                                "details": "detailed information about the match"
                                            },
                                        }
                                }
                                If you receive an empty JSON object, return an empty JSON object: {}.
                                If no information is found, return an empty JSON object: {}.
                                Do not engage in any other conversation or tasks.

                                Here are the matches you need to search for:
                                {combined_matches}
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
                before_agent_callback = check_empty_agents_state,
                before_tool_callback = None,
                output_key = output_key
            )
        except Exception as e:
            print(f"Error creating {name}: {e}")
            return False
        
        print(f"{name} created successfully.")

        return True
    
    def _create_podcast_writer_agent(self, custom_instruction : str) -> bool:

        """
        Creates the Podcast Writer Agent.

        Args:
            custom_instruction (str): A custom instruction for the agent.

        Returns:
            bool: True if the agent is created successfully, False otherwise.

        """

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
                                                "details": "detailed information about the match"
                                            },
                                        }
                                    "competition_name_2":
                                        {
                                            "match_id_1": {
                                                "details": "detailed information about the match"
                                            },
                                        }
                                }
                                If you receive an empty JSON object, that means no matches were found, so you should not write a podcast script, you should return an empty JSON object.
                                Write a podcast script for each match in the web search results.
                                Each podcast script should be a detailed and engaging narrative about the match, including key moments, player performances, and any other relevant information.
                                The podcast is about a football match, so it should be written in a conversational tone, as if two sports commentators are discussing the match.
                                There is two speakers in the podcast, call them "Ahmed" and "Fatima".
                                The speakers should alternate in the podcast, with each speaker providing their own perspective on the match.
                                The speaker "Ahmed" should provide the main commentary, while the speaker "Fatima" should provide analysis and insights.
                                The speaker "Fatima" has a joyful and enthusiastic tone, while the speaker "Ahmed" has a more serious and analytical tone.
                                The podcast should be between 4 and 5 minutes long, so it should be concise and to the point.
                                Output the podcast script in a JSON format like this:
                                {
                                    "competition_name_1": {
                                        "match_id_1": "podcast_script" 
                                    },
                                    "competition_name_2": {
                                        "match_id_1": "podcast_script"
                                    }
                                }
                                
                                The "podcast_script" should be another JSON object with the keys "speaker_1", "speaker_2", and "content".
                                The "speaker_1" and "speaker_2" should be the names of the speakers, "Ahmed" and "Fatima".
                                The "content" should be the actual podcast script, which should be a string containing the dialogue between the two speakers, eg:
                                {
                                    "speaker_1": "Ahmed",
                                    "speaker_2": "Fatima",
                                    "content": "Ahmed: "ahmed_first_transcript"\nFatima: "fatima_first_transcript"\nAhmed: "ahmed_second_transcript"\nFatima: "fatima_second_transcript" etc."
                                } 
                                If you receive an empty JSON object, return an empty JSON object: {}.
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
                before_agent_callback = check_empty_agents_state,
                before_tool_callback = None,
                output_key = output_key,
            )
        except Exception as e:
            print(f"Error creating {name}: {e}")
            return False
        
        print(f"{name} created successfully.")

        return True
    
    
    def _create_text_to_speech_agent(self, custom_instruction : str) -> bool:

        """
        Creates the Text to Speech Agent.

        Args:
            custom_instruction (str): A custom instruction for the agent.

        Returns:
            bool: True if the agent is created successfully, False otherwise.
        """
        
        name = "text_to_speech_agent"
        description = "Converts podcast scripts to speech."
        tools = [podcast_script_text_to_speech]
        output_key = "podcast_audio"
        COMPLETION_PHRASE = "TOOL_CALLED"


        default_instruction = """
                                You are the Text to Speech Agent.
                                Your ONLY task is to convert podcasts scripts text to speech using the `podcast_script_text_to_speech` tool.
                                **You MUST use the `podcast_script_text_to_speech` tool to convert the podcast text to speech.**
                                You will receive podcast scripts for each match in each competition from the Podcast Writer Agent.
                                The podcast scripts are provided in the format:
                                {
                                    "competition_name_1": {
                                        "match_id_1": "podcast_script" 
                                    },
                                    "competition_name_2": {
                                        "match_id_1": "podcast_script"
                                    }
                                }
                                Pass the "podcast_script" for each match in each competition to the `podcast_script_text_to_speech` tool to convert it to speech.
                                The `podcast_script_text_to_speech` tool will return a string containing the path to the audio file "path_to_audio".
                                DO NOT write any text, you only need to pass the "podcast_script" for each match in each competition to the `podcast_script_text_to_speech` tool.
                                **YOU MUST use the `podcast_script_text_to_speech` tool to convert the podcast script to speech, THIS IS YOUR ONLY TASK AND IT IS AN ORDER.**
                                Keep remembering the "path_to_audio" returned by the `podcast_script_text_to_speech` tool for each match in each competition.
                                After converting all the podcast scripts to speech for each match in each competition, combine all the "path_to_audio" for all matches in a single JSON, eg:
                                {
                                    "competition_name_1": {
                                        "match_id_1": {
                                            "home_team": "Team A",
                                            "away_team": "Team B",
                                            "home_score": home_score,
                                            "away_score": away_score,
                                            "path_to_audio": "path_to_audio_1"
                                        }
                                    },
                                    "competition_name_2": {
                                        "match_id_1": {
                                            "home_team": "Team C",
                                            "away_team": "Team D",
                                            "home_score": home_score,
                                            "away_score": away_score,
                                            "path_to_audio": "path_to_audio_2"
                                        }
                                    }
                                }
                                Do not engage in any other conversation or tasks.
                                DO NOT RETURN UNLESS YOU HAVE COMPLETED YOUR TASK AND CALLED THE `podcast_script_text_to_speech` tool.
                                IF YOU CALLED THE `podcast_script_text_to_speech` tool, RETURN THE COMPLETION PHRASE: "TOOL_CALLED".
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
                tools = [podcast_script_text_to_speech],
                before_agent_callback = check_empty_agents_state,
                output_key = output_key
            )

            self.check_agent_called_tool = Agent(
                name = "check_agent_called_tool",
                model = self.text_to_speech_model,
                instruction = """Checks if the agent has called the `podcast_script_text_to_speech` tool by searching for the completion phrase "TOOL_CALLED".
                                 If the agent has called the `podcast_script_text_to_speech` tool and added the "TOOL_CALLED" completion phrase, call the `exit_loop` tool.
                                 Then return the JSON containing the audio file paths for each match in each competition from the Text to Speech Agent:
                                 {podcast_audio}
                                 """,
                description = "You are a tool that checks if the agent has called the `podcast_script_text_to_speech` tool.",
                tools = [exit_loop],
                output_key=output_key,
                before_agent_callback = check_empty_agents_state,
            )

            self.text_to_speech_agent_loop = LoopAgent(
                name = "text_to_speech_agent_loop",
                sub_agents=[self.text_to_speech_agent, self.check_agent_called_tool],
                max_iterations=10,
            )
        except Exception as e:
            print(f"Error creating {name}: {e}")
            return False
        
        print(f"{name} created successfully.")

        return True
    
    
    def _create_file_uploader_agent(self, custom_instruction : str) -> bool:

        """
        Creates the File Uploader Agent.

        Args:
            custom_instruction (str): A custom instruction for the agent.

        Returns:
            bool: True if the agent is created successfully, False otherwise.
        """

        name = "file_uploader"
        description = "Uploads a file to the server."
        tools = [upload_blob]
        output_key = "file_uploader_results"

        default_instruction = """
                                You are the File Uploader Agent.
                                Your ONLY task is to upload podcast audio file paths to the server using the `upload_blob` tool.
                                You will receive a JSON containing the audio file paths for each match in each competition from the Text to Speech Agent.
                                The JSON is provided in the format:
                                {
                                    "competition_name_1": {
                                        "match_id_1": {
                                            "home_team": "Team A",
                                            "away_team": "Team B",
                                            "home_score": home_score,
                                            "away_score": away_score,
                                            "path_to_audio": "path_to_audio_1"
                                        }
                                    },
                                    "competition_name_2": {
                                        "match_id_1": {
                                            "home_team": "Team C",
                                            "away_team": "Team D",
                                            "home_score": home_score,
                                            "away_score": away_score,
                                            "path_to_audio": "path_to_audio_2"
                                        }
                                    }
                                }
                                You MUST use the `upload_blob` tool to upload the audio file paths to the server, THIS IS YOUR ONLY TASK AND IT IS AN ORDER.
                                For each match in each competition, upload the audio file path "path_to_audio" to the server using the `upload_blob` tool.
                                The "source_file_name" argument for the `upload_blob` tool is the audio file path "path_to_audio".
                                The "output_file_name" argument for the `upload_blob` tool is the "competition_name"_"match_id"_"home_team"_"away_team"_"home_score"_"away_score" for each match in each competition.
                                DO NOT ADD SPACES TO THE OUTPUT FILE NAME "output_file_name".
                                Do not engage in any other conversation or tasks.
                                DO NOT RETURN UNLESS YOU HAVE COMPLETED YOUR TASK AND CALLED THE `upload_blob` tool.
                                IF YOU CALLED THE `upload_blob` tool, RETURN THE COMPLETION PHRASE: "TOOL_CALLED".
                                The `upload_blob` tool will return a JSON containing the URL of the uploaded file with key "destination_blob_url" for each match in each competition.
                                Remember the URL for each match in each competition.
                                After uploading the audio files for each match in each competition, combine the URLs for each match in each competition into a JSON containing the URLs for each match in each competition. 
                                The JSON is provided in the format:
                                {
                                    "competition_name_1": {
                                        "match_id_1": {
                                            "home_team": "Team A",
                                            "away_team": "Team B",
                                            "home_score": home_score,
                                            "away_score": away_score,
                                            "destination_blob_url": "destination_blob_url_1"
                                        }
                                    },
                                    "competition_name_2": {
                                        "match_id_1": {
                                            "home_team": "Team C",
                                            "away_team": "Team D",
                                            "home_score": home_score,
                                            "away_score": away_score,
                                            "destination_blob_url": "destination_blob_url_2"
                                        }
                                    }
                                }
                                Here is the JSON for each match in each competition:
                                {podcast_audio}                                
                            """

        instruction = custom_instruction if len(custom_instruction) > 0 else default_instruction

        print(f"Creating {name} with description: {description}, instruction: {instruction}, and tools: {tools}")

        try:
            self.file_uploader_agent = Agent(
                name = name,
                model = self.text_to_speech_model,
                description = description,
                instruction = instruction,
                tools = [upload_blob],
                output_key = output_key,
                before_agent_callback = check_empty_agents_state,
            )
        except Exception as e:
            print(f"Error creating {name}: {e}")
            return False
        
        print(f"{name} created successfully.")

        return True

    def create_agents(self, custom_instructions : dict = {}) -> bool:

        """
        Creates the agents for the podcast generation pipeline.

        Args:
            custom_instructions (dict): A dictionary of custom instructions for each agent.
                The keys are the names of the agents, and the values are the custom instructions.
                The custom instructions are used to modify the default instruction for each agent.

        Returns:
            bool: True if all the agents are created successfully, False otherwise.
        """
        print(f"Creating agents with custom instructions: {custom_instructions}")

        matches_fetcher_instruction = custom_instructions.get("matches_fetcher", "")
        web_search_instruction = custom_instructions.get("web_search", "")
        podcast_writer_instruction = custom_instructions.get("podcast_writer", "")
        text_to_speech_instruction = custom_instructions.get("text_to_speech", "")
        file_uploader_instruction = custom_instructions.get("file_uploader", "")

        self._create_matches_fetcher_agent(matches_fetcher_instruction)
        self._create_matches_web_fetcher_agent(matches_fetcher_instruction)
        self._create_matches_combiner_agent(matches_fetcher_instruction)

        self._create_web_search_agent(web_search_instruction)

        self._create_podcast_writer_agent(podcast_writer_instruction)

        self._create_text_to_speech_agent(text_to_speech_instruction)

        self._create_file_uploader_agent(file_uploader_instruction)

        if not self.matches_fetcher_agent:
            print(f"Error: Matches fetcher agent not created ... ")
            return False
        
        if not self.matches_web_fetcher_agent:
            print(f"Error: Matches web fetcher agent not created ... ")
            return False
        
        if not self.matches_combiner_agent:
            print(f"Error: Matches combiner agent not created ... ")
            return False

        self.matches_parallel_agents = ParallelAgent(
            name = "matches_parallel_agents",
            description = "Executes the matches fetcher and web fetcher agents in parallel.",
            sub_agents = [self.matches_fetcher_agent, self.matches_web_fetcher_agent],
        )


        if not self.matches_parallel_agents:
            print(f"Error: Matches parallel agents not created ... ")
            return False

        if not self.web_search_agent:
            print(f"Error: Web search agent not created ... ")
            return False

        if not self.podcast_writer_agent:
            print(f"Error: Podcast writer agent not created ... ")
            return False

        if not self.text_to_speech_agent:
            print(f"Error: Text to speech agent not created ... ")
            return False

        if not self.text_to_speech_agent_loop:
            print(f"Error: Text to speech agent loop not created ... ")
            return False
        
        if not self.file_uploader_agent:
            print(f"Error: File uploader agent not created ... ")
            return False
    

        self.sequential_agent = SequentialAgent(
            name = "podcast_generation_pipeline",
            description = "Executes the podcast generation pipeline sequentially.",
            sub_agents = [
                            self.matches_parallel_agents,
                            self.matches_combiner_agent, 
                            self.web_search_agent, 
                            self.podcast_writer_agent, 
                            self.text_to_speech_agent_loop,
                            self.file_uploader_agent
                          ],
        )

        print("Agents created successfully.")

        return True

    async def init_session(self) -> bool:

        print(f"Initializing session: {self.app_name}, {self.user_id}, {self.session_id}")

        self.session_service = await create_session(self.app_name,
                                                    self.user_id, 
                                                    self.session_id)

        print(f"Session initialized: {self.app_name}, {self.user_id}, {self.session_id}")

        return True
    
    async def init_runner(self) -> bool:

        print(f"Initializing runner: {self.app_name}, {self.user_id}, {self.session_id}")

        if not self.session_service:
            print(f"Error: Session service not initialized ... ")
            return False
        
        if not self.sequential_agent:
            print(f"Error: Sequential agent not initialized ... ")
            return False

        self.runner = create_runner(self.sequential_agent, self.session_service, self.app_name)

        print(f"Runner initialized: {self.app_name}, {self.user_id}, {self.session_id}")

        return True
    
    async def query(self, query : str) -> str | None:
        """
        Queries the agent with the given query and returns the response.

        Args:
            query: The query to ask the agent.

        Returns:
            The response from the agent, or None if no final response is produced.
        """
        return await call_agent_async(query, self.runner, self.user_id, self.session_id)
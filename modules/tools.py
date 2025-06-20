import os
import requests

from datetime import datetime
from typing import Dict, Union
from dotenv import load_dotenv

from google.adk.tools.tool_context import ToolContext
from google.adk.tools import google_search  # Import the tool
from google import genai
from google.genai import types

from modules.utils import wave_file


load_dotenv()

print(f"FOOTBALL_DATA_API Key set: {'Yes' if os.environ.get('FOOTBALL_DATA_API_KEY') and os.environ['FOOTBALL_DATA_API_KEY'] != 'FOOTBALL_DATA_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")
print(f"Google API Key set: {'Yes' if os.environ.get('GOOGLE_API_KEY') and os.environ['GOOGLE_API_KEY'] != 'YOUR_GOOGLE_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")


MODEL_GEMINI_2_5_FLASH_PREVIEW_TTS = "gemini-2.5-flash-preview-tts"

def get_matches_by_date(date_str: str , tool_context: ToolContext) -> dict:

    """
        Fetches all matches for a given date from the Football Data API.

        Args:
            date_str (str): The date to fetch matches for in the format YYYY-MM-DD (e.g. "2023-05-31").
        Returns:
           dict: A dictionary with a status key set to "success" in case matches are found, "error" in case no matches are found. 
                In case of matches are found, the result will contain a dictionary with match ID as the key,
                and a dictionary containing the competition, home team, away team, home score and away score as the value if matches are found,
                or an error message if no matches are found.
    """ 

    agent_name = ""

    if tool_context is not None:
        agent_name = tool_context.agent_name

    tool_name = "get_matches_by_date"

    result: Dict[str, Union[str, Dict]] = {"status": "success"}

    print(f"--- Tool : {tool_name} called for date: {date_str} by agent: {agent_name} ---")

    date = datetime.now()
    current_date = datetime.now().date()
    
    if date_str is not None:
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
    
    if date > current_date:
        return {"error": f"Date must not be in the future, provided date: {date}, current date: {current_date}"}
        
    date_formatted = date.strftime("%Y-%m-%d")

    print(f"--- Tool : {tool_name} Fetching matches for date: {date_formatted}, current date: {current_date}")

    api_key = os.getenv("FOOTBALL_DATA_API_KEY")

    uri = f"https://api.football-data.org/v4/matches/?date={date_formatted}"

    print(f"--- Tool : {tool_name} URI: {uri}")

    headers = { 'X-Auth-Token': api_key }

    response = requests.get(uri, headers=headers)

    response_json = response.json()

    error_code = response_json.get("errorCode", -1)

    if error_code > 0:

        result["status"] = "error"
        result["error"] = f"API error code : {error_code}, response: {response_json}"

        return result

    matches = response_json.get("matches", [])

    if len(matches) == 0:

        result["status"] = "error"
        result["error"] = f"No matches found for date: {date}"

        return result
    
    matches_dict = {}

    for match in matches:

        competition = match["competition"]["name"]

        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]

        score = match["score"]["fullTime"]

        home_score = score["home"]
        away_score = score["away"]

        matches_dict[str(match["id"])] = {
            "competition": competition,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score
        }

    print(f"--- Tool : {tool_name} Matches found: {matches_dict} ---")

    result["matches"] = matches_dict

    return result


def podcast_text_to_speech(podcast_script: dict, tool_context: ToolContext) -> str:

    """
        Converts a podcast script (transcript of a conversation) to speech.

        Args:
            podcast_script (dict): A dictionary containing the transcript of the conversation to be converted to speech, with keys "speaker_1", "speaker_2", and "content".
        Returns:
            str: The path of the saved audio file containing the generated speech.
    """ 

    agent_name = ""

    if tool_context is not None:
        agent_name = tool_context.agent_name

    tool_name = "podcast_text_to_speech"

    print(f"--- Tool : {tool_name} called by agent: {agent_name} ---")

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    speaker_1 = podcast_script.get("speaker_1", "Joe")
    speaker_2 = podcast_script.get("speaker_2", "Jane")

    content = podcast_script.get("content", "")

    prompt = f"""TTS the following conversation between {speaker_1} and {speaker_2}:
           {content}"""
    
    print(f"--- Tool : {tool_name} Generating audio for prompt: {prompt} ---")

    response = client.models.generate_content(

        model=MODEL_GEMINI_2_5_FLASH_PREVIEW_TTS,
        contents=prompt,

        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],

            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(

                    speaker_voice_configs=[

                        types.SpeakerVoiceConfig(
                            speaker=speaker_1,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name='Kore',
                                )
                            )
                        ),
                        
                        types.SpeakerVoiceConfig(
                            speaker=speaker_2,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name='Puck',
                                )
                            )
                        ),
                    ]
                )
            )
        )
    )

    if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts or not response.candidates[0].content.parts[0].inline_data:
        print(f"--- Tool : {tool_name} No audio content generated ---")
        return "No audio content generated"
    
    data = response.candidates[0].content.parts[0].inline_data.data

    if not os.path.exists("output"):
        os.makedirs("output")

    file_name= f"output/out.wav"
    if tool_context is not None:
        file_name = f"output/{tool_context.agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    wave_file(file_name, data) # Saves the file to current directory

    print(f"--- Tool : {tool_name} Audio file saved as: {file_name} ---")

    return file_name  # Return the name of the saved audio file
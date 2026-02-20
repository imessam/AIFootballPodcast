import os
import sys
import requests

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from datetime import datetime
from typing import Dict, Optional, Union
from dotenv import load_dotenv

import asyncio
import edge_tts
from modules.utils import wave_file


load_dotenv()

print(f"SPORT_DEV_KEY Key set: {'Yes' if os.environ.get('SPORT_DEV_KEY') and os.environ['SPORT_DEV_KEY'] != 'SPORT_DEV_KEY' else 'No (REPLACE PLACEHOLDER!)'}")


def get_matches_by_date(date_str: str, leagues_id: list) -> dict:

    """
        Fetches all matches for a given date from the Sportdevs API.

        Args:
            date_str (str): The date to fetch matches for in the format YYYY-MM-DD (e.g. "2023-05-31").
        Returns:
           dict: A dictionary with a status key set to "success" in case matches are found, "error" in case no matches are found. 
                In case of matches are found, the result will contain a dictionary with match ID as the key,
                and a dictionary containing the competition, home team, away team, home score and away score as the value if matches are found,
                or an error message if no matches are found.
    """ 
    tool_name = "get_matches_by_date"

    result: Dict[str, Union[str, Dict]] = {"status": "success"}

    print(f"--- Tool : {tool_name} called for date: {date_str} ---")

    date = datetime.now()
    current_date = datetime.now().date()
    
    if date_str is not None:
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
    
    if date > current_date:
        return {"error": f"Date must not be in the future, provided date: {date}, current date: {current_date}"}
        
    date_formatted = date.strftime("%Y-%m-%d")

    print(f"--- Tool : {tool_name} Fetching matches for date: {date_formatted}, current date: {current_date}")

    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    # Football-Data.org uri for matches
    uri = f"https://api.football-data.org/v4/matches?dateFrom={date_formatted}&dateTo={date_formatted}"

    print(f"--- Tool : {tool_name} URI: {uri} ---")

    headers = { 'X-Auth-Token': f"{api_key}" }

    response = requests.get(uri, headers=headers)
    
    if response.status_code != 200:
        print(f"--- Tool : {tool_name} Error: API returned status {response.status_code} ---")
        print(f"--- Raw Error Response: {response.text} ---")
        return {"status": "error", "error": f"API returned status {response.status_code}", "matches": {}}

    try:
        response_json = response.json()
        print(f"--- Tool : {tool_name} Raw API Response Successfully Logged ---")
        # Log limited version of the response to avoid cluttering but show structure
        # print(f"--- Raw JSON: {response_json} ---") 
    except Exception as e:
        print(f"--- Tool : {tool_name} Error parsing JSON: {e} ---")
        return {"status": "error", "error": "Invalid JSON response", "matches": {}}

    return response_json


async def local_text_to_speech(text: str, speaker_name: str = "en-US-ChristopherNeural") -> str:
    """
    Converts text to speech using a local TTS (edge-tts).

    Args:
        text (str): The text to convert to speech.
        speaker_name (str): The name of the voice to use. Defaults to "en-US-ChristopherNeural".

    Returns:
        str: The path to the generated audio file.
    """
    tool_name = "local_text_to_speech"
    print(f"--- Tool : {tool_name} Generating audio for text: {text[:50]}... ---")

    out_dir = "output"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    file_name = f"{out_dir}/podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    communicate = edge_tts.Communicate(text, speaker_name)
    await communicate.save(file_name)

    print(f"--- Tool : {tool_name} Audio file saved as: {file_name} ---")

    return file_name



if __name__ == "__main__":
    import asyncio
    # Simple test for local tts
    async def test():
        await local_text_to_speech("Testing local podcast generation.")
    asyncio.run(test())
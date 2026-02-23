import os
import re
from typing import Annotated, TypedDict, List, Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from modules.tools import get_matches_by_date, local_text_to_speech, search_football_news
from modules.tts import TTSManager
from modules.utils import wave_file
from modules.constants import DEFAULT_COMPETITIONS

# Define the State
class AgentState(TypedDict):
    query: str
    matches: Dict[str, Any]
    news: List[str]
    script: str
    audio_path: str
    errors: List[str]

class FootballPodcastAgent:
    """
    Encapsulates the logic for fetching matches, generating a script,
    and converting it to audio using LangGraph.
    """

    def __init__(self):
        self.llm = self._get_llm()
        self.graph = self._create_podcast_graph()

    def _get_llm(self):
        local_base_url = os.getenv("LOCAL_OPENAI_BASE_URL", "http://localhost:11434/v1")
        local_model = os.getenv("LOCAL_MODEL_NAME", "qwen3:0.6b")
        return ChatOpenAI(
            model=local_model,
            openai_api_key="none",
            base_url=local_base_url,
        )

    # Node 1: Fetch Matches
    def fetch_matches_node(self, state: AgentState):
        print("--- [FootballPodcastAgent] Node: fetch_matches_node ---")
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            result = get_matches_by_date(today, DEFAULT_COMPETITIONS)
            matches_list = result.get("matches", [])
            print(f"--- [FootballPodcastAgent] Matches Fetched: {len(matches_list)} ---")
            return {"matches": result, "errors": []}
        except Exception as e:
            print(f"--- [FootballPodcastAgent] Error in fetch_matches_node: {e} ---")
            return {"errors": [f"Error fetching matches: {str(e)}"]}

    # Node 2: Search Web for News
    def search_news_node(self, state: AgentState):
        print("--- [FootballPodcastAgent] Node: search_news_node ---")
        matches_data = state.get("matches", {})
        matches_list = matches_data.get("matches", [])
        
        if not matches_list:
            return {"news": ["No matches found today."]}
        
        all_news = []
        for match in matches_list:
            home = match.get('homeTeam', {}).get('name', 'Unknown')
            away = match.get('awayTeam', {}).get('name', 'Unknown')
            score = f"{match.get('score', {}).get('fullTime', {}).get('home', '?')}-{match.get('score', {}).get('fullTime', {}).get('away', '?')}"
            
            match_summary = f"Match: {home} vs {away}. Result: {score}."
            all_news.append(match_summary)
            
            # Fetch recent news related to this match
            query = f"{home} vs {away} football news"
            news_snippets = search_football_news(query, max_results=3)
            if news_snippets:
                all_news.append(f"Latest news around {home} vs {away}:\n{news_snippets}")
        
        return {"news": all_news}

    # Node 3: Generate Script
    def generate_script_node(self, state: AgentState):
        print("--- [FootballPodcastAgent] Node: generate_script_node ---")
        context = "\n".join(state.get("news", []))

        prompt = f"""
You are writing a script for a football podcast hosted by two presenters:
  ALEX  — the authoritative lead anchor, calm and analytical.
  JAMIE — the enthusiastic co-host, energetic and opinionated.

Write a lively, engaging dialogue of at least 400 words between ALEX and JAMIE
covering today's football matches. The script should:
  1. Open with a warm intro from ALEX.
  2. Have JAMIE jump in with excitement.
  3. Cover EACH match in depth — discuss the scoreline, key moments,
     standout players, and tactical observations, trading lines back
     and forth between ALEX and JAMIE.
  4. Include opinions, hypotheticals, and light banter.
  5. Close with a sign-off from both hosts.

If no matches are listed, discuss the upcoming week in football and
what fans should look forward to.

Match data:
{context}

Format EVERY line as:
ALEX: <what Alex says>
JAMIE: <what Jamie says>

Do NOT include any stage directions, sound effects, markdown formatting,
bold/italic markers, or anything other than the ALEX:/JAMIE: lines.
Wrap the entire script in <script> tags.
"""
        
        messages = [
            SystemMessage(content="You are a football podcast writer. You provide concise match summaries."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content

            print(f"--- [FootballPodcastAgent] Raw Script Content: {content} ---")

            # re is already imported at module level
            # Step 1: Strip <think>...</think> reasoning blocks first (Qwen / reasoning models)
            content_clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            # Step 2: Try to extract content inside <script>...</script> tags
            script_match = re.search(r'<script[^>]*>(.*?)</script>', content_clean, re.DOTALL | re.IGNORECASE)
            if script_match:
                final_script = script_match.group(1).strip()
            else:
                # Fallback: strip any remaining <script> tags and use the whole clean text
                final_script = re.sub(r'</?script[^>]*>', '', content_clean, flags=re.IGNORECASE).strip()

            if not final_script:
                final_script = content_clean or content

            print(f"--- [FootballPodcastAgent] Script Generated Successfully ---")
            return {"script": final_script}
        except Exception as e:
            print(f"--- [FootballPodcastAgent] Error in generate_script_node: {e} ---")
            return {"script": "", "errors": state.get("errors", []) + [f"Error generating script: {str(e)}"]}

    @staticmethod
    def _clean_segment_for_tts(text: str) -> str:
        """Clean a single speaker segment for TTS (no speaker labels present)."""
        # Remove any stray XML/HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove Markdown bold/italic markers
        text = re.sub(r'[*_]{1,3}(.*?)[*_]{1,3}', r'\1', text)
        # Remove parenthetical stage directions like (laughs) or (pause)
        text = re.sub(r'\(.*?\)', '', text)
        # Collapse extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def _parse_script_segments(script: str) -> list:
        """
        Parse a two-host dialogue script into (speaker, text) tuples.

        Expected format (one turn per line):
            ALEX: Welcome to the show...
            JAMIE: Thanks Alex...

        Returns a list of (speaker_name, cleaned_text) tuples.
        Lines that don't match the pattern are attached to the previous speaker.
        """
        segments = []
        current_speaker = None
        current_lines = []

        pattern = re.compile(r'^([A-Z][A-Z0-9 _-]*):\s*(.*)', re.IGNORECASE)

        for line in script.splitlines():
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                # Save the previous turn first
                if current_speaker and current_lines:
                    segments.append((current_speaker, ' '.join(current_lines)))
                current_speaker = m.group(1).strip().upper()
                current_lines = [m.group(2).strip()] if m.group(2).strip() else []
            else:
                # Continuation line for the current speaker
                if current_lines is not None:
                    current_lines.append(line)

        # Flush last turn
        if current_speaker and current_lines:
            segments.append((current_speaker, ' '.join(current_lines)))

        return segments

    # Node 4: Text to Speech
    async def tts_node(self, state: AgentState):
        print("--- [FootballPodcastAgent] Node: tts_node ---")
        script = state.get("script", "")
        if not script:
            return {"errors": ["No script available for TTS."]}

        # Parse into (speaker, text) segments
        segments = self._parse_script_segments(script)

        if not segments:
            # Fallback: no recognisable speaker labels — treat as single voice
            print("--- [FootballPodcastAgent] No speaker labels found; falling back to single-voice TTS. ---")
            clean = self._clean_segment_for_tts(script)
            try:
                audio_path = await local_text_to_speech(clean)
                return {"audio_path": audio_path}
            except Exception as e:
                print(f"--- [FootballPodcastAgent] Error in fallback TTS: {e} ---")
                return {"errors": [f"Error in TTS: {str(e)}"]}

        # Clean each segment individually
        clean_segments = [
            (speaker, self._clean_segment_for_tts(text))
            for speaker, text in segments
            if text.strip()
        ]

        print(f"--- [FootballPodcastAgent] Generating dialogue TTS for {len(clean_segments)} segments ---")
        for i, (spk, txt) in enumerate(clean_segments):
            print(f"  [{i+1}] {spk}: {txt[:80]}...")

        try:
            audio_path = await TTSManager.generate_audio_dialogue(clean_segments)
            return {"audio_path": audio_path}
        except Exception as e:
            print(f"--- [FootballPodcastAgent] Error in dialogue TTS: {e} ---")
            return {"errors": [f"Error in dialogue TTS: {str(e)}"]}

    # Build the Graph
    def _create_podcast_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("fetch_matches", self.fetch_matches_node)
        workflow.add_node("search_news", self.search_news_node)
        workflow.add_node("generate_script", self.generate_script_node)
        workflow.add_node("tts", self.tts_node)

        workflow.add_edge(START, "fetch_matches")
        workflow.add_edge("fetch_matches", "search_news")
        workflow.add_edge("search_news", "generate_script")
        workflow.add_edge("generate_script", "tts")
        workflow.add_edge("tts", END)

        return workflow.compile()

    async def run(self, query: str):
        """Runs the agent graph with the given query."""
        return await self.graph.ainvoke({"query": query})

# Maintain legacy creator for compatibility if needed
def create_podcast_graph():
    return FootballPodcastAgent().graph

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    import asyncio
    agent = FootballPodcastAgent()
    
    asyncio.run(run_example())

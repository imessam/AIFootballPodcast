import os
from typing import Annotated, TypedDict, List, Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from modules.tools import get_matches_by_date, local_text_to_speech
from modules.utils import wave_file

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
            result = get_matches_by_date(today, [27])
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
            all_news.append(f"Match: {home} vs {away}. Result: {score}.")
        
        return {"news": all_news}

    # Node 3: Generate Script
    def generate_script_node(self, state: AgentState):
        print("--- [FootballPodcastAgent] Node: generate_script_node ---")
        context = "\n".join(state.get("news", []))
        
        prompt = f"""
        Write a short, exciting podcast script summarizing today's football matches.
        If matches are listed below, summarize the results or upcoming fixtures.
        If no matches are listed, mention that it's a quiet day in football.
        
        Matches Info:
        {context}
        
        The script should be conversational and professional. 
        Wrap the final script in <script> tags.
        """
        
        messages = [
            SystemMessage(content="You are a football podcast writer. You provide concise match summaries."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content
            
            import re
            script_match = re.search(r'<script>(.*?)</script>', content, re.DOTALL)
            if script_match:
                final_script = script_match.group(1).strip()
            else:
                final_script = re.sub(r'<think>.*?(?:</think>|$)', '', content, flags=re.DOTALL).strip()
                final_script = re.sub(r'</?script>', '', final_script).strip()
            
            if not final_script:
                final_script = content
            
            print(f"--- [FootballPodcastAgent] Script Generated Successfully ---")
            return {"script": final_script}
        except Exception as e:
            print(f"--- [FootballPodcastAgent] Error in generate_script_node: {e} ---")
            return {"script": "", "errors": state.get("errors", []) + [f"Error generating script: {str(e)}"]}

    # Node 4: Text to Speech
    async def tts_node(self, state: AgentState):
        print("--- [FootballPodcastAgent] Node: tts_node ---")
        script = state.get("script", "")
        if not script:
            return {"errors": ["No script available for TTS."]}
        
        try:
            audio_path = await local_text_to_speech(script)
            return {"audio_path": audio_path}
        except Exception as e:
            print(f"--- [FootballPodcastAgent] Error in local TTS: {e} ---")
            return {"errors": [f"Error in local TTS: {str(e)}"]}

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

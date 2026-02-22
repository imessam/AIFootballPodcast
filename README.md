# AIFootballPodcast
AIFootballPodcast leverages LangGraph as its agent framework, utilizes local LLM models for large language model tasks, text-to-speech (TTS), and employs the Google Search API for web research. The system orchestrates a pipeline of specialized agents to automate the creation of football match podcasts, from data collection and research to script generation, audio synthesis, and cloud publishing.
The project is part of [Google's ADK hackathon](https://googlecloudmultiagents.devpost.com/).


## Agents Flow Diagram.
![AIFootballPodcastDiagram](assets/diagram.png)

## Models used:

- Local LLM for text generation: Ollama (qwen3:0.6b, etc)
- Chatterbox TTS for podcast script text to speech: [Chatterbox](https://github.com/chatterbox-tts/chatterbox)

## Web UI live demo deployed on Google Cloud run.

- URL: [https://football-podcast-agent-388890953707.us-central1.run.app](https://football-podcast-agent-388890953707.us-central1.run.app)

## Youtube Demo Video.

[![FootballPodcastDemo](https://img.youtube.com/vi/hOF1Z-7p4qY/0.jpg)](https://www.youtube.com/watch?v=hOF1Z-7p4qY)

## Features

- Fetches football match data from the Football Data API and Google Search.
- Performs web research on each match using the Google Search API.
- Generates podcast scripts for each match using local LLMs.
- Converts scripts to speech using Chatterbox TTS.
- Uploads generated audio files to Google Cloud Storage.
- Modular agent-based architecture using LangGraph.

## Project Structure

```
AIFootballPodcast/

├── modules/         # Core modules for data fetching, processing, and synthesis
├── output/          # Output directory for audio files
├── run.py           # Entry point for running agents with custom queries
├── .env             # Environment variables
├── deploy.sh        # Deployment script
├── requirements.txt # Python dependencies
└── README.md        # Project documentation
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/imessam/AIFootballPodcast.git
    cd AIFootballPodcast
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up required API keys and credentials for:
    - Football Data API
4. API keys and credentials should be set as environment variables or in a `.env` file:

    ```bash
    export FOOTBALL_DATA_API_KEY="YOUR_FOOTBALL_DATA_API_KEY"
    export LOCAL_OPENAI_BASE_URL="http://localhost:11434/v1"
    export LOCAL_MODEL_NAME="qwen3:0.6b"
    ```
## Usage

- To run the agents with custom queries:
  ```bash
  python run.py {query} 
  Examples:

  Podcast about a specific match: 
  python run.py "Ahly vs Real Madrid"

  Podcast from a specific date:
  python run.py "2025-05-31"
  ```

## Issues

- Sometimes, the text to speech agent doesn't call the `podcast_text_to_speech` tool. This is related to the instructions given to the agent.
- If you find any issues or have suggestions, please open an issue on GitHub.

## TODO
- [ ] Support other models for LLM and TTS, open or closed sources.
- [ ] Support multiple languages.
- [ ] Longer and more informative podcast scripts.
- [ ] Different podcast speakers voices and characters.
- [ ] Generate animated podcasts.

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).

## Acknowledgements


- Local LLM via Ollama and Chatterbox TTS
- Google Search API
- Football Data API


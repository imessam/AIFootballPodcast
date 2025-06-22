# AIFootballPodcast
AIFootballPodcast leverages Google's Agent Development Kit (ADK) as its agent framework, utilizes Gemini models for both large language model (LLM) tasks and text-to-speech (TTS) synthesis, and employs the Google Search API for web research. The system orchestrates a pipeline of specialized agents to automate the creation of football match podcasts, from data collection and research to script generation, audio synthesis, and cloud publishing.

## Live demo deployed on Google Cloud run.

- URL: [https://football-podcast-agent-388890953707.us-central1.run.app](https://football-podcast-agent-388890953707.us-central1.run.app)

## Youtube Demo Video.

- URL: [https://www.youtube.com/](https://www.youtube.com)

## Features

- Fetches football match data from the Football Data API and Google Search.
- Performs web research on each match using the Google Search API.
- Generates podcast scripts for each match using Gemini LLMs.
- Converts scripts to speech using Gemini TTS.
- Uploads generated audio files to Google Cloud Storage.
- Modular agent-based architecture using Google's ADK.

## Project Structure

```
AIFootballPodcast/
├── agent.py         # ADK Web UI agent definitions
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
    - Google Gemini API KEY
    - Football Data API
4. API keys and credentials should be set as environment variables or in a `.env` file:

    ```bash
    export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    export FOOTBALL_DATA_API_KEY="YOUR_FOOTBALL_DATA_API_KEY"
    export GOOGLE_GENAI_USE_VERTEXAI=FALSE
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
- To use the ADK Web UI, run:
  ```bash
  adk web
  ```

## Issues

- Sometimes, the text to speech agent doesn't call the `podcast_text_to_speech` tool. This is related to the instructions given to the agent.
- If you find any issues or have suggestions, please open an issue on GitHub.

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).

## Acknowledgements

- Google Agent Development Kit (ADK)
- Gemini LLM and TTS
- Google Search API
- Football Data API


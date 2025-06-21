import os
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

import vertexai
import uuid

from vertexai import agent_engines

from dotenv import load_dotenv

from modules.agents_podcast import PodcastAgents

load_dotenv()

# --- Verify Keys (Optional Check) ---
print("API Keys Set:")
print(f"Google API Key set: {'Yes' if os.environ.get('GOOGLE_API_KEY') and os.environ['GOOGLE_API_KEY'] != 'YOUR_GOOGLE_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")
print(f"GOOGLE_GENAI_USE_VERTEXAI Key set: {'Yes' if os.environ.get('GOOGLE_GENAI_USE_VERTEXAI') and os.environ['GOOGLE_GENAI_USE_VERTEXAI'] else 'No (REPLACE PLACEHOLDER!)'}")


PROJECT_ID = "gen-lang-client-0894170450"
LOCATION = "us-central1"
STAGING_BUCKET = "gs://gemini_podacst_agent_bucket"

requirements = "requirements.txt"
extra_packages = ["modules"]

env_vars = {
  "GOOGLE_GENAI_USE_VERTEXAI": os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", ""),
  "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY", ""),
  "FOOTBALL_DATA_API_KEY": os.environ.get("FOOTBALL_DATA_API_KEY", ""),
}

gcs_dir_name = str(uuid.uuid4())
gcs_dir = f"gemini_podacst_agent/{gcs_dir_name}"

display_name = "Football Podcast Agent"

description = """
This is a football podcast agent.
"""



print(
    f"""
        Using 
        Project ID: {PROJECT_ID}, 
        Location: {LOCATION}, 
        Staging Bucket: {STAGING_BUCKET}, 
        GCS Directory: {gcs_dir},
        Environment Variables: {env_vars},
        Requirements: {requirements},
        Extra Packages: {extra_packages},
        Display Name: {display_name},
        Description: {description}
    """)


vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=STAGING_BUCKET,
)


podcast_agent_object = PodcastAgents()

# Create the agents.
podcast_agent_object.create_agents()

root_agent = podcast_agent_object.sequential_agent


remote_agent = agent_engines.create(
    root_agent,                    # Optional.
    requirements=requirements,      # Optional.
    extra_packages=extra_packages,  # Optional.
    gcs_dir_name=gcs_dir_name,      # Optional.
    display_name=display_name,      # Optional.
    description=description,        # Optional.
    env_vars=env_vars,              # Optional.
)


print(f"Remote agent created with resource name: {remote_agent.resource_name}")









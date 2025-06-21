#! /bin/bash

# Set your Google Cloud Project ID
export GOOGLE_CLOUD_PROJECT="gen-lang-client-0894170450"

# Set your desired Google Cloud Location
export GOOGLE_CLOUD_LOCATION="us-central1" # Example location

# Set the path to your agent code directory
export AGENT_PATH="." # Assuming capital_agent is in the current directory

# Set a name for your Cloud Run service (optional)
export SERVICE_NAME="football-podcast-agent"

# Set an application name (optional)
export APP_NAME="football-podcast-agent"

# export VOLUME_NAME="football-podcast-volume"
# export BUCKET_NAME="football-podcast-bucket"
# export MOUNT_PATH="/mnt/football-podcast"

adk deploy cloud_run \
--project=$GOOGLE_CLOUD_PROJECT \
--region=$GOOGLE_CLOUD_LOCATION \
--service_name=$SERVICE_NAME \
--app_name=$APP_NAME \
--with_ui \
$AGENT_PATH

# gcloud run services update $SERVICE_NAME \
# --add-volume name=VOLUME_NAME,type=cloud-storage,bucket=BUCKET_NAME \
# --add-volume-mount volume=VOLUME_NAME,mount-path=MOUNT_PATH
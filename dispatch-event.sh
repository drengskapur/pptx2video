#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Usage: ./send_dispatch_event.sh <repository> <event_type> [<client_payload_json>]
# Example: ./send_dispatch_event.sh owner/repo "event-type" '{"key": "value"}'

REPOSITORY="$1"
EVENT_TYPE="$2"
CLIENT_PAYLOAD="$3"

# Check if GH_TOKEN is set
if [ -z "$GH_TOKEN" ]; then
  echo "Error: GH_TOKEN environment variable is not set."
  exit 1
fi

# Validate inputs
if [ -z "$REPOSITORY" ] || [ -z "$EVENT_TYPE" ]; then
  echo "Usage: $0 <repository> <event_type> [<client_payload_json>]"
  exit 1
fi

# Prepare data payload
if [ -n "$CLIENT_PAYLOAD" ]; then
  DATA=$(jq -n \
    --arg event_type "$EVENT_TYPE" \
    --argjson client_payload "$CLIENT_PAYLOAD" \
    '{event_type: $event_type, client_payload: $client_payload}')
else
  DATA=$(jq -n \
    --arg event_type "$EVENT_TYPE" \
    '{event_type: $event_type}')
fi

# Send the repository dispatch event
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: Bearer $GH_TOKEN" \
  https://api.github.com/repos/"$REPOSITORY"/dispatches \
  -d "$DATA")

# Check response
if [ "$RESPONSE" -eq 204 ]; then
  echo "Repository dispatch event '$EVENT_TYPE' sent successfully to $REPOSITORY."
else
  echo "Failed to send repository dispatch event. HTTP status code: $RESPONSE"
  exit 1
fi

#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="$SCRIPT_DIR"
DEMO_URL="http://127.0.0.1:5000"
HEALTH_URL="$DEMO_URL/health"

fail_demo() {
  print -u2 -- "Demo error: $1"
  exit 1
}

cd "$PROJECT_ROOT" || fail_demo "Cannot open the project directory."

command -v uv >/dev/null 2>&1 || fail_demo "uv is not installed or is not on PATH."
command -v curl >/dev/null 2>&1 || fail_demo "curl is required to open the demo safely."
command -v open >/dev/null 2>&1 || fail_demo "The macOS open command is unavailable."

print -- "Preparing the locked offline demo environment..."
if ! uv sync --locked; then
  fail_demo "uv sync --locked failed."
fi

print -- "Verifying the packaged model, health endpoint, and prediction path..."
if ! uv run arxiv-classifier --root "$PROJECT_ROOT" smoke >/dev/null; then
  fail_demo "Smoke verification failed."
fi

(
  for attempt in {1..80}; do
    if curl --fail --silent --show-error "$HEALTH_URL" >/dev/null 2>&1; then
      open "$DEMO_URL" || print -u2 -- "Demo error: Could not open the default browser. Visit $DEMO_URL manually."
      exit 0
    fi
    sleep 0.1
  done
  print -u2 -- "Demo error: The local server did not become ready."
) &

print -- "Serving the offline demo at $DEMO_URL"
print -- "Keep this Terminal window open. Press Control-C to stop the server."
exec uv run arxiv-classifier --root "$PROJECT_ROOT" serve \
  --host 127.0.0.1 \
  --port 5000 \
  --llm-mode offline

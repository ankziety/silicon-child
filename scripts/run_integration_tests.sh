#!/usr/bin/env bash
set -euo pipefail

if [ "${ENVIRONMENT:-development}" != "development" ]; then
  echo "Integration tests may only be run in development environment. Set ENVIRONMENT=development and retry."
  exit 1
fi

if [ -z "${LLMZ_API_KEY:-}" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "No aggregator API keys found. Set LLMZ_API_KEY or OPENROUTER_API_KEY to run integration tests."
  exit 1
fi

echo "Running integration tests..."
pytest tests/integration -q



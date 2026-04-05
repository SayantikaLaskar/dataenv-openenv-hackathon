#!/usr/bin/env sh
set -eu

SPACE_URL="${1:-http://127.0.0.1:7860}"
PROJECT_DIR="${2:-.}"

echo "Checking openenv.yaml"
test -f "$PROJECT_DIR/openenv.yaml"

echo "Checking inference.py"
test -f "$PROJECT_DIR/inference.py"

echo "Checking health endpoint"
curl -fsS "$SPACE_URL/health" >/dev/null

echo "Checking reset endpoint"
curl -fsS -X POST "$SPACE_URL/reset" >/dev/null

echo "Basic submission checks passed"

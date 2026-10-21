#!/bin/zsh
# Load tests for three llama-server setups behind a local gateway with escalation OFF.
# Usage: loadtest/run_all.sh   (stops any running llama-server first; ~45 min)
cd "${0:a:h}/.."
MODEL=models/storeops-v2-q8_0.gguf
ESCALATION_ENABLED=false LLM_MODEL=storeops-v2-q8_0 RATE_LIMIT_PER_MIN=100000 \
  uv run uvicorn serve.gateway.app:app --port 8010 --log-level warning > data/loadtest-gateway.log 2>&1 &
GW=$!
trap 'kill $GW; pkill -f "llama-server -m"' EXIT
run() {  # label, ngl, slots, users...
  local label=$1 ngl=$2 slots=$3; shift 3
  pkill -f "llama-server -m"; sleep 2
  llama-server -m $MODEL --alias storeops-v2-q8_0 --jinja -ngl $ngl -c $(( 8192 * slots )) -np $slots \
    --metrics --host 127.0.0.1 --port 8080 > data/llama-server.log 2>&1 &
  for i in {1..90}; do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 1; done
  uv run python -m loadtest.loadtest --label $label --users "$@" --seconds 120
}
run metal-1slot 99 1 1 2 4 8 16
run metal-4slots 99 4 1 2 4 8 16
run cpu-2slots 0 2 1 2 4

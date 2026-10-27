#!/bin/zsh
# Load tests for three llama-server setups behind a local gateway with escalation OFF.
# Usage: loadtest/run_all.sh   (the served 3B; stops any running llama-server first; ~45 min)
#        MODEL=models/storeops-v2-q8_0.gguf ALIAS=storeops-v2-q8_0 PREFIX=1.5b- loadtest/run_all.sh
cd "${0:a:h}/.."
MODEL=${MODEL:-models/storeops-3b-q4_k_m.gguf}
ALIAS=${ALIAS:-storeops-3b-q4_k_m}
PREFIX=${PREFIX:-3b-}
# Simulated users stage orders: use a scratch copy, never the reference store DB.
mkdir -p data/loadtest && cp data/store.db data/loadtest/store.db
STORE_DB_PATH=data/loadtest/store.db ESCALATION_ENABLED=false LLM_MODEL=$ALIAS RATE_LIMIT_PER_MIN=100000 \
  uv run uvicorn serve.gateway.app:app --port 8010 --log-level warning > data/loadtest-gateway.log 2>&1 &
GW=$!
trap 'kill $GW; pkill -f "[l]lama-server -m"' EXIT
run() {  # label, ngl, slots, users...
  local label=$1 ngl=$2 slots=$3; shift 3
  pkill -f "[l]lama-server -m"; sleep 2
  llama-server -m $MODEL --alias $ALIAS --jinja -ngl $ngl -c $(( 8192 * slots )) -np $slots \
    --metrics --host 127.0.0.1 --port 8080 > data/llama-server.log 2>&1 &
  for i in {1..90}; do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 1; done
  uv run python -m loadtest.loadtest --label $PREFIX$label --users "$@" --seconds 120
}
run metal-1slot 99 1 1 2 4 8 16
run metal-4slots 99 4 1 2 4 8 16
run cpu-2slots 0 2 1 2 4

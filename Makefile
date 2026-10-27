# One command starts everything: llama-server natively on the Mac (Metal), the rest in Docker.
# Served model: the 3B student (93.4% with the gateway, all local). The 1.5B: MODEL=models/storeops-v2-q8_0.gguf ALIAS=storeops-v2-q8_0
MODEL ?= models/storeops-3b-q4_k_m.gguf
ALIAS ?= storeops-3b-q4_k_m
# 4 slots doubled throughput vs 1 in the load test
SLOTS ?= 4
LLM_LOG = data/llama-server.log

.PHONY: up down llm llm-stop docker reset-db logs test lint ps

up: llm docker  ## start everything
	@echo "UI http://localhost:8000 | API http://localhost:8000/docs | Grafana http://localhost:3000 | Prometheus http://localhost:9090"

llm:  ## llama-server on Metal (all layers on GPU), metrics on
	@curl -sf http://127.0.0.1:8080/health >/dev/null && echo "llama-server already running" || \
	 (nohup llama-server -m $(MODEL) --alias $(ALIAS) --jinja -ngl 99 -c $$(( 8192 * $(SLOTS) )) -np $(SLOTS) \
	   --metrics --host 127.0.0.1 --port 8080 > $(LLM_LOG) 2>&1 & \
	  for i in $$(seq 1 60); do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 1; done; \
	  echo "llama-server up ($(MODEL), $(SLOTS) slots)")

docker:  ## the running app writes orders to its own copy of the store DB, not data/store.db
	@mkdir -p data/runtime && [ -f data/runtime/store.db ] || cp data/store.db data/runtime/store.db
	docker compose up -d --build

reset-db:  ## start the app over with a fresh copy of the store DB
	rm -f data/runtime/store.db && cp data/store.db data/runtime/store.db

down: llm-stop  ## stop everything
	docker compose down

llm-stop:
	-pkill -f "[l]lama-server -m" || true

logs:
	docker compose logs -f gateway

ps:
	docker compose ps

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

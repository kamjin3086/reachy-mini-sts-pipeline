PYTHON ?= python3
STS_PYTHON ?= /home/kamjin/apps/.venv-qwen3-fa/bin/python3

.PHONY: help start start-fast verify-flash bench bench-llm test clean-local

help:
	@printf '%s\n' \
		'Common commands:' \
		'  make start        Start the stable STS path' \
		'  make start-fast   Start Qwen3-TTS FastAPI + flash-attn path' \
		'  make verify-flash Verify the isolated flash-attn environment' \
		'  make bench        Run quick end-to-end benchmark' \
		'  make bench-llm    Run LLM TTFT benchmark' \
		'  make test         Run repository tests' \
		'  make clean-local  Remove local caches/logs generated in this repo'

start:
	./scripts/sts_start.sh

start-fast:
	./scripts/sts_start_qwen3_openai_fastapi_flash.sh

verify-flash:
	env -u HSA_OVERRIDE_GFX_VERSION $(STS_PYTHON) scripts/verify_qwen3_flash_attn_env.py

bench:
	$(PYTHON) scripts/bench_sts_pipeline.py --quick

bench-llm:
	$(PYTHON) scripts/bench_llm_models.py

test:
	$(PYTHON) -m pytest tests

clean-local:
	find scripts -type d -name __pycache__ -prune -exec rm -rf {} +
	find tests -type d -name __pycache__ -prune -exec rm -rf {} +
	find scripts -type f -name 'log_*.txt' -delete

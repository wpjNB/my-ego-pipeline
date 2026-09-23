# Convenience targets. Everything here is optional sugar over the scripts.
SHELL := /bin/bash
CONDA ?= conda
ENV ?= ego3d_base
CLIP ?= demo01
CONFIG ?= configs/macrodata_final.yaml

.PHONY: help env install test test-fast dry-run clean smoke

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

env: ## create the orchestrator environment (CPU-friendly)
	$(CONDA) env create -f environment-base.yml

install: ## editable install of the orchestrator package
	$(CONDA) run -n $(ENV) python -m pip install --no-build-isolation -e .

test: ## run the full test suite in the base env
	$(CONDA) run -n $(ENV) python -m pytest -q

test-fast: ## run everything except the synthetic stitching scene
	$(CONDA) run -n $(ENV) python -m pytest -q -k "not stitch"

smoke: ## preprocess a synthetic video end to end through Phase 0
	bash scripts/smoke_test.sh

dry-run: ## print the pipeline plan without running the model backends
	$(CONDA) run -n $(ENV) python scripts/run_pipeline.py \
		--config $(CONFIG) --clip $(CLIP) --video data/raw/$(CLIP).mp4 --dry-run

clean: ## remove caches (never touches data/, weights/ or third_party/)
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

.DEFAULT_GOAL := help
SHELL := /bin/bash

UV ?= uv
FROM ?= 2024-03-01
TO   ?= 2024-03-15
Q    ?= Welke uitspraken gaan over verduistering door een penningmeester?

.PHONY: help install lint fmt test ci ingest index ask eval demo

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create .venv and install the package with dev tools
	$(UV) sync --all-extras

lint:  ## ruff check + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:  ## ruff format + autofix
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

test:  ## Unit tests: fixtures + fakes, no network, no model, no LLM (PRD N1)
	$(UV) run pytest

ci: lint test  ## Exactly what the CI workflow runs, locally

# ---------------------------------------------------------------- the chain
ingest:  ## Fetch raw XML for the period FROM..TO into data/raw/ (resumable)
	$(UV) run strafrecht-rag ingest --from-date $(FROM) --to-date $(TO)

index:  ## Parse data/raw/, chunk, embed and upsert into the vector store
	$(UV) run strafrecht-rag index

ask:  ## Ask a question: make ask Q="..."
	$(UV) run strafrecht-rag ask "$(Q)"

eval:  ## Recall@k, source correctness and no-source correctness on the eval set
	$(UV) run strafrecht-rag eval

demo:  ## Animated walkthrough of the whole chain (for recordings)
	$(UV) run strafrecht-rag demo

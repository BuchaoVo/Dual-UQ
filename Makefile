.PHONY: install check init test lint
install:
	pip install -e ".[dev]"

check:
	python scripts/00_check_environment.py

init:
	python scripts/01_init_manifests.py

test:
	pytest -q

lint:
	ruff check src scripts tests


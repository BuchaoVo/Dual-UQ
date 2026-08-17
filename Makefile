.PHONY: install check init test lint
install:
	pip install -e ".[dev]"

check:
	python scripts/maintenance/check_environment.py

init:
	python -c "from dual_uq.manifests import initialize_manifests; initialize_manifests('data/manifests')"

test:
	pytest -q

lint:
	ruff check src scripts tests

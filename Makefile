.PHONY: install test lint typecheck check clean

PY ?= python3

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -v

lint:
	$(PY) -m ruff check folding_astar tests

typecheck:
	$(PY) -m mypy folding_astar

# Run everything CI runs.
check: lint typecheck test

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

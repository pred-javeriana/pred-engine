.PHONY: test test-cov lint format pyright

test:
	uv run pytest -q

test-cov:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

pyright:
	uv run pyright

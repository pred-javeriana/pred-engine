.PHONY: test test-cov lint format pyright motor-local

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

# Verificacion local del motor antes de pedir revision; ver docs/features/motor-local/README.md.
# Ejemplo: make motor-local ARGS="--publicar"
motor-local:
	uv run python .github/scripts/motor_local.py $(ARGS)

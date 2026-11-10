.PHONY: setup test test-cov lint format pyright motor-local

# Instala las dependencias fijadas y deja en la cache los pesos de Chronos-2 (familia foundation),
# asi la primera corrida no los descarga. Sin espacio o memoria para Chronos-2:
# make setup FUNDACIONAL=no, y corra sin foundation (--no-foundation).
FUNDACIONAL ?= si
setup:
ifeq ($(FUNDACIONAL),no)
	uv sync --frozen --extra dev
else
	uv sync --frozen --extra dev --extra foundation
	uv run python -c "from pred_engine.comun.modelos.modelos_fundacionales import CHRONOS2_ZERO_SHOT, cargar_pipeline; cargar_pipeline(CHRONOS2_ZERO_SHOT)"
endif

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

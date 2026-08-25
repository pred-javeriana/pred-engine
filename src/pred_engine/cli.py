"""CLI de operador para ingesta 1.2 y consulta del catalogo LLM."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pred_engine.comun.llm import (
    DEFAULT_MODELS,
    LlmProviderError,
    LlmTimeoutError,
    UnknownModelError,
    UnknownProviderError,
    build_llm_provider,
    format_models_help,
    normalize_provider_name,
    resolve_model,
)
from pred_engine.comun.logger import configure_json_logger, get_logger
from pred_engine.ingesta.lector import extract_csv
from pred_engine.ingesta.sonda import SemanticAlignmentError, probe_headers

_logger = get_logger(__name__)

_ENV_KEY = {
    "gemini": ("PRED_LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("PRED_LLM_API_KEY", "OPENAI_API_KEY"),
    "anthropic": ("PRED_LLM_API_KEY", "ANTHROPIC_API_KEY"),
}


def resolve_api_key(provider: str, explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    canonico = normalize_provider_name(provider)
    for nombre in _ENV_KEY[canonico]:
        valor = os.environ.get(nombre)
        if valor and valor.strip():
            return valor.strip()
    raise ValueError(
        "Falta API key. Pase --api-key o defina PRED_LLM_API_KEY / "
        + " / ".join(_ENV_KEY[canonico][1:])
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pred-engine",
        description=(
            "PRED engine — ingesta 1.2 (sonda consultiva + esquema + remuestreo)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    modelos = sub.add_parser(
        "models",
        help="Lista modelos permitidos por proveedor (tier economico)",
    )
    modelos.add_argument(
        "--provider",
        default="gemini",
        help="gemini | openai | anthropic (alias: google, gpt, claude)",
    )

    probe = sub.add_parser(
        "probe",
        help="Diagnostico LLM de cabeceras sin mutar el CSV ni ejecutar el pipeline",
    )
    probe.add_argument(
        "--csv", required=True, type=Path, help="Ruta al CSV del operador"
    )
    probe.add_argument(
        "--provider",
        default="gemini",
        help="gemini | openai | anthropic (alias: google, gpt, claude)",
    )
    probe.add_argument("--api-key", default=None, help="Clave LLM (no se registra)")
    probe.add_argument(
        "--model",
        default=None,
        help=(
            "Modelo del catalogo del proveedor. "
            f"Default: tier economico ({DEFAULT_MODELS}). "
            "Use 'pred-engine models --provider X' para ver la lista."
        ),
    )
    probe.add_argument("--data-root", type=Path, default=Path("data"))
    probe.add_argument("--timeout", type=float, default=30.0)

    ingest = sub.add_parser(
        "ingest", help="Diagnosticar, validar y exportar Parquet si la sonda acepta"
    )
    ingest.add_argument(
        "--csv", required=True, type=Path, help="Ruta al CSV del operador"
    )
    ingest.add_argument(
        "--provider",
        default="gemini",
        help="gemini | openai | anthropic (alias: google, gpt, claude)",
    )
    ingest.add_argument("--api-key", default=None, help="Clave LLM (no se registra)")
    ingest.add_argument(
        "--model",
        default=None,
        help=(
            "Modelo del catalogo del proveedor. "
            f"Default: tier economico ({DEFAULT_MODELS}). "
            "Use 'pred-engine models --provider X' para ver la lista."
        ),
    )
    ingest.add_argument("--data-root", type=Path, default=Path("data"))
    ingest.add_argument("--timeout", type=float, default=30.0)
    return parser


def _build_provider(args: argparse.Namespace):
    clave = resolve_api_key(args.provider, args.api_key)
    canonico = normalize_provider_name(args.provider)
    modelo = resolve_model(canonico, args.model)
    proveedor = build_llm_provider(canonico, clave, model=modelo)
    return canonico, modelo, proveedor


def _cmd_models(provider: str) -> int:
    canonico = normalize_provider_name(provider)
    defecto = DEFAULT_MODELS[canonico]
    print(f"Proveedor: {canonico}")
    print(f"Default (mas barato): {defecto}")
    print("Modelos permitidos:")
    print(format_models_help(canonico))
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    try:
        canonico, modelo, proveedor = _build_provider(args)
    except LlmProviderError as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 1
    except (UnknownProviderError, UnknownModelError, ValueError) as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 1

    from pred_engine.ingesta.pipeline import deposit_raw_csv

    try:
        crudo = deposit_raw_csv(args.csv, data_root=args.data_root)
        extraido = extract_csv(crudo, data_root=args.data_root)
        artefacto = probe_headers(extraido.frame, proveedor, timeout=args.timeout)
    except SemanticAlignmentError as exc:
        _logger.error("%s", exc)
        print("estado: rejected")
        print("diagnostico:", exc.diagnostic_json())
        print("columnas_intactas:", list(extraido.frame.columns))
        print("filas_muestra:", extraido.row_count)
        return 0
    except LlmTimeoutError as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 4
    except LlmProviderError as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError) as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 1

    print("estado: accepted")
    print("proveedor:", canonico)
    print("modelo:", modelo)
    print("diagnostico:", artefacto.diagnostic.model_dump_json(ensure_ascii=False))
    print("columnas_intactas:", list(artefacto.frame.columns))
    print("filas_muestra:", extraido.row_count)
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    try:
        canonico, modelo, proveedor = _build_provider(args)
    except LlmTimeoutError as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 4
    except LlmProviderError as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 1
    except (
        UnknownProviderError,
        UnknownModelError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 1

    from pred_engine.ingesta.pipeline import run_ingest
    from pred_engine.ingesta.validador_formato import SchemaBarrierError

    try:
        resultado = run_ingest(
            args.csv,
            proveedor,
            data_root=args.data_root,
            timeout=args.timeout,
        )
    except SemanticAlignmentError as exc:
        _logger.error("%s", exc)
        print(exc.diagnostic_json(), file=sys.stderr)
        return 2
    except SchemaBarrierError as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 3
    except LlmTimeoutError as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 4
    except LlmProviderError as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 1
    except (
        UnknownProviderError,
        UnknownModelError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 1

    print("proveedor:", canonico)
    print("modelo:", modelo)
    print(
        "diagnostico:",
        resultado.diagnostic.diagnostic.model_dump_json(ensure_ascii=False),
    )
    print("columnas_intactas:", list(resultado.diagnostic.frame.columns))
    print("filas_crudas:", resultado.source.row_count)
    print("filas_validadas:", len(resultado.validated))
    print("filas_panel_diario:", len(resultado.panel))
    print("parquet:", resultado.parquet_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_json_logger("pred_engine")
    args = build_parser().parse_args(argv)
    if args.command == "models":
        try:
            return _cmd_models(args.provider)
        except UnknownProviderError as exc:
            _logger.error("%s", exc)
            print(exc, file=sys.stderr)
            return 1
    if args.command == "probe":
        return _cmd_probe(args)
    if args.command == "ingest":
        return _cmd_ingest(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

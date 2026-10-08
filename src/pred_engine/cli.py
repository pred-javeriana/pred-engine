"""Diagnosticos de operador para ingesta y el pipeline backend L1-L4."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pred_engine.aumentacion.fase0 import (
    ERRORES_FASE0,
    agregar_opciones_fase0,
    configuracion_desde_argumentos,
    ejecutar_fase_0,
)
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
from pred_engine.ingesta.categorizacion import (
    TopologyArtifact,
    TopologyContractError,
    TopologyMathError,
    TopologyRoutingError,
)
from pred_engine.ingesta.lector import extract_csv
from pred_engine.ingesta.salida import OutputHandoffError
from pred_engine.ingesta.sonda import SemanticAlignmentError, probe_headers

if TYPE_CHECKING:
    from pred_engine.pipeline import Pipeline, PipelineInput, PipelineResult

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


def _imprimir_topologia(artefacto: TopologyArtifact) -> None:
    print("sku_id,n_periods,n_positive,adi,cv2,sku_class")
    for m in artefacto.metrics:
        print(
            f"{m.sku_id},{m.n_periods},{m.n_positive},"
            f"{m.adi:.6f},{m.cv2:.6f},{m.sku_class}"
        )
    resumen: dict[str, int] = {}
    for m in artefacto.metrics:
        resumen[m.sku_class] = resumen.get(m.sku_class, 0) + 1
    print("sku_class_resumen:", resumen)


def _imprimir_contrato(marco, parquet: Path) -> None:
    print("contrato_1_4: accepted")
    print("parquet:", parquet)
    print("filas:", len(marco))
    print("skus:", int(marco["sku_id"].nunique()))
    print("columnas:", ",".join(str(c) for c in marco.columns))
    por_sku = (
        marco.drop_duplicates("sku_id")["sku_class"]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    print("sku_class_resumen_sku:", por_sku)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pred-engine",
        description=(
            "PRED engine — ingesta 1.2, topologia 1.3 y contrato de salida 1.4."
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
        "ingest",
        help="Diagnosticar, validar, remuestrear, clasificar, validar 1.4 y publicar",
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

    classify = sub.add_parser(
        "classify",
        help="Motor 1.3: ADI/CV2/sku_class sobre CSV canonico o Parquet 1.2 (sin LLM)",
    )
    classify.add_argument(
        "--csv", type=Path, default=None, help="CSV con cabeceras canonicas"
    )
    classify.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Parquet de panel diario (salida 1.2)",
    )
    classify.add_argument("--data-root", type=Path, default=Path("data"))

    verify = sub.add_parser(
        "verify",
        help="Releer un Parquet 1.4 y validar el contrato (sin LLM)",
    )
    verify.add_argument(
        "--parquet",
        required=True,
        type=Path,
        help="Parquet publicado en processed/",
    )
    run = sub.add_parser(
        "run",
        help="Ejecutar M0 (opcional) y el pipeline backend L1-L4",
        description=(
            "Corre L1-L3 y persiste la corrida en {runs-dir}/{run_id}. Con "
            "--seed-csv primero genera el panel sintetico de la Fase 0 (opciones "
            "--m0-*) y lo usa como entrada de L1. Codigos de salida: 7 = L4 "
            "ausente sin fallos; 8 = L4 ausente con unidades fallidas aisladas; "
            "1 = una etapa sin resultados."
        ),
    )
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path)
    source.add_argument("--parquet", type=Path, help="Artefacto clasificado 1.4")
    source.add_argument(
        "--seed-csv", type=Path, help="Semilla de la Fase 0 (M0 -> M1 -> M2)"
    )
    run.add_argument("--data-root", type=Path, default=Path("data"))
    run.add_argument("--provider", default=None, help="Sonda semantica opcional")
    run.add_argument("--api-key", default=None)
    run.add_argument("--model", default=None)
    run.add_argument("--timeout", type=float, default=30.0)
    run.add_argument(
        "--families",
        nargs="+",
        default=["classical", "ml", "dl"],
        choices=["classical", "ml", "dl", "foundation"],
    )
    run.add_argument("--trials", type=int, default=None)
    run.add_argument("--min-train", type=int, default=40)
    run.add_argument("--horizon", type=int, default=7)
    run.add_argument("--step", type=int, default=7)
    run.add_argument("--seasonality", type=int, default=7)
    run.add_argument(
        "--metric", choices=["mae", "rmse", "smape", "mase"], default="rmse"
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Procesos para las unidades SKU x familia (por defecto, nucleos - 1)",
    )
    run.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Directorio de corridas (por defecto {data-root}/runs)",
    )
    run.add_argument(
        "--run-id",
        default=None,
        help="Identidad de la corrida; por defecto se deriva de entrada y opciones",
    )
    run.add_argument(
        "--telemetry-interval",
        type=float,
        default=1.0,
        help=(
            "Segundos entre muestras de CPU y memoria por proceso en "
            "recursos.jsonl; 0 desactiva el muestreo"
        ),
    )
    m0 = run.add_argument_group("Fase 0 (solo con --seed-csv)")
    agregar_opciones_fase0(m0, prefijo="m0-")

    telemetry = sub.add_parser(
        "telemetry",
        help="Graficar CPU, memoria y la traza de modelos de una corrida terminada",
        description=(
            "Lee unidades.jsonl, recursos.jsonl y corrida.json de una corrida y "
            "escribe un SVG con etapas, CPU, memoria (total y por proceso) y una "
            "fila por proceso con cada unidad SKU x familia coloreada por modelo."
        ),
    )
    telemetry.add_argument("run_dir", type=Path, help="Directorio de la corrida")
    telemetry.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta del SVG (por defecto {run_dir}/telemetria.svg)",
    )
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
    except OutputHandoffError as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 6
    except (TopologyMathError, TopologyRoutingError, TopologyContractError) as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 5
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
    _imprimir_topologia(resultado.topology)
    _imprimir_contrato(resultado.panel, resultado.parquet_path)
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    from pred_engine.ingesta.pipeline import run_classify_csv, run_classify_parquet
    from pred_engine.ingesta.validador_formato import SchemaBarrierError

    if (args.csv is None) == (args.parquet is None):
        print("Pase exactamente uno de --csv o --parquet", file=sys.stderr)
        return 1
    try:
        if args.csv is not None:
            topologia, destino = run_classify_csv(args.csv, data_root=args.data_root)
        else:
            topologia, destino = run_classify_parquet(
                args.parquet, data_root=args.data_root
            )
    except SchemaBarrierError as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 3
    except OutputHandoffError as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 6
    except (TopologyMathError, TopologyRoutingError, TopologyContractError) as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 5
    except (ValueError, FileNotFoundError) as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 1

    print("filas_panel_diario:", len(topologia.frame))
    _imprimir_topologia(topologia)
    _imprimir_contrato(topologia.frame, destino)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from pred_engine.ingesta.pipeline import run_verify_parquet

    try:
        marco = run_verify_parquet(args.parquet)
    except OutputHandoffError as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 6
    except FileNotFoundError as exc:
        _logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 1

    _imprimir_contrato(marco, Path(args.parquet).expanduser().resolve())
    return 0


def _exit_code(result: PipelineResult) -> int:
    if result.complete:
        return 0
    return 8 if result.failures else 7


def diagnose_pipeline(
    pipeline: Pipeline,
    request: PipelineInput,
) -> tuple[int, dict[str, Any]]:
    """Technical adapter only; the application service owns all stage transitions."""
    from pred_engine.pipeline import summarize_run

    code, result, failed_stage = _run_pipeline(pipeline, request)
    summary = summarize_run(result)
    if failed_stage is not None:
        summary["failed_stage"] = failed_stage
    return code, summary


def _run_pipeline(
    pipeline: Pipeline, request: PipelineInput
) -> tuple[int, PipelineResult, str | None]:
    from pred_engine.pipeline import PipelineExecutionError

    try:
        result = pipeline.run(request)
    except PipelineExecutionError as exc:
        return 1, exc.result, exc.stage
    return _exit_code(result), result, None


_RUN_SETTINGS = (
    "families",
    "trials",
    "min_train",
    "horizon",
    "step",
    "seasonality",
    "metric",
    "seed",
    "provider",
    "model",
)


def _cmd_run(args: argparse.Namespace) -> int:
    from pred_engine.comun.ejecucion_paralela import resumen_paralelismo
    from pred_engine.comun.reserva import RESERVE_FRACTION
    from pred_engine.optimizacion.router import PredictorFamily
    from pred_engine.pipeline import (
        EvaluationSettings,
        PipelineInput,
        summarize_run,
    )
    from pred_engine.pipeline_setup import build_pipeline
    from pred_engine.run_artifacts import (
        code_fingerprint,
        library_versions,
        run_identifier,
        sha256_file,
        write_run,
    )
    from pred_engine.run_telemetry import ResourceSampler

    if args.telemetry_interval < 0:
        print("error: --telemetry-interval debe ser >= 0", file=sys.stderr)
        return 1
    csv_path = args.csv
    inputs: dict[str, Any] = {}
    try:
        if args.seed_csv is not None:
            fase0 = ejecutar_fase_0(
                args.seed_csv,
                configuracion_desde_argumentos(args, prefijo="m0-"),
                data_root=args.data_root,
                reutilizar=True,
            )
            csv_path = fase0.artefacto.path
            inputs["m0"] = {
                "seed_csv": str(Path(args.seed_csv).expanduser().resolve()),
                "artifact": str(fase0.artefacto.path),
                "rows": fase0.artefacto.row_count,
                "sha256": fase0.artefacto.sha256,
                "run_log": str(fase0.bitacora_path),
                "reused": fase0.reutilizada,
            }
        source = Path(csv_path if csv_path is not None else args.parquet)
        inputs["source"] = {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
        }
        settings = {name: getattr(args, name) for name in _RUN_SETTINGS} | {
            "reserve_fraction": RESERVE_FRACTION,
            "pred_engine": library_versions().get("pred-engine", "desconocida"),
            "code": code_fingerprint(),
        }
        run_id = args.run_id or run_identifier(inputs["source"]["sha256"], settings)
        runs_dir = args.runs_dir or Path(args.data_root) / "runs"
        run_dir = Path(runs_dir) / run_id
        provider = _build_provider(args)[2] if args.provider is not None else None
        pipeline = build_pipeline(
            EvaluationSettings(
                min_train=args.min_train,
                horizon=args.horizon,
                step=args.step,
                seasonality=args.seasonality,
                metric=args.metric,
            ),
            families=cast(tuple[PredictorFamily, ...], tuple(args.families)),
            n_trials=args.trials,
            seed=args.seed,
            workers=args.workers,
            hpo_root=run_dir / "hpo",
            session=run_id,
        )
        request = PipelineInput(
            csv_path=csv_path,
            parquet_path=args.parquet,
            data_root=args.data_root,
            provider=provider,
            timeout=args.timeout,
        )
        sampler = (
            ResourceSampler(run_dir / "recursos.jsonl", args.telemetry_interval)
            if args.telemetry_interval > 0
            else None
        )
    except ERRORES_FASE0 + (OSError, LlmProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    started = time.time()
    with sampler or nullcontext():
        code, result, failed_stage = _run_pipeline(pipeline, request)
    metadata = {
        "inputs": inputs,
        "settings": settings | {"workers": args.workers},
        "wall_seconds": round(time.time() - started, 3),
        "telemetry": sampler.summary() if sampler is not None else None,
    }
    if result.ingestion is not None:
        metadata["inputs"]["m1"] = {
            "parquet": str(result.ingestion.parquet_path),
            "rows": len(result.ingestion.panel),
            "skus": len(set(result.ingestion.panel["sku_id"].tolist())),
            "sha256": sha256_file(result.ingestion.parquet_path),
        }
    try:
        files = write_run(
            result,
            run_dir,
            run_id=run_id,
            metadata=metadata,
            resources=sampler.path if sampler is not None else None,
        )
    except (OSError, ValueError) as exc:
        print(
            f"error: no se pudo persistir la corrida {run_id}: {exc}", file=sys.stderr
        )
        return 1
    summary = summarize_run(result)
    if failed_stage is not None:
        summary["failed_stage"] = failed_stage
    summary["run_id"] = run_id
    summary["run_dir"] = str(run_dir)
    summary["files"] = sorted(path.name for path in files.values())
    summary["wall_seconds"] = metadata["wall_seconds"]
    summary["parallelism"] = {
        stage: resumen_paralelismo([u.record for u in result.units if u.stage == stage])
        for stage in ("L2", "L3")
    }
    print(json.dumps(summary, ensure_ascii=False))
    return code


def _cmd_telemetry(args: argparse.Namespace) -> int:
    from pred_engine.run_chart import render_run_chart

    try:
        path = render_run_chart(args.run_dir, args.output)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(path)
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
    if args.command == "classify":
        return _cmd_classify(args)
    if args.command == "verify":
        return _cmd_verify(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "telemetry":
        return _cmd_telemetry(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

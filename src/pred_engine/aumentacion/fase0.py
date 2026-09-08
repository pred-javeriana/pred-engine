"""0.4-C1 - Script orquestador de la Fase 0 (One-Shot Execution).

Encadena, en una unica invocacion y sin acoplarse al Framework PRED (Modulo 1):

    semilla -> remuestreo (MBB + rejection sampling)
            -> compuerta de restricciones fisicas
            -> validador de conformidad de esquema
            -> exportador CSV bajo politica WORM

La semilla aleatoria se propaga a todas las etapas estocasticas para que la
corrida sea reproducible bit a bit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pred_engine.aumentacion.bitacora import (
    BitacoraCorrida,
    persistir_bitacora,
)
from pred_engine.aumentacion.conformidad import validar_conformidad_o_fallar
from pred_engine.aumentacion.contrato import CONTRACT_VERSION, OUTPUT_COLUMNS
from pred_engine.aumentacion.divergencia import (
    TOLERANCIA_DIVERGENCIA_POR_DEFECTO,
)
from pred_engine.aumentacion.exportador_csv import (
    MINIMO_FILAS_POR_DEFECTO,
    NOMBRE_ARTEFACTO_POR_DEFECTO,
    ArtefactoExportado,
    exportar_artefacto_csv,
)
from pred_engine.aumentacion.rechazo import generar_series_aceptadas
from pred_engine.aumentacion.restricciones import (
    aplicar_restricciones_fisicas,
    limites_lead_time_desde_semilla,
)
from pred_engine.comun.logger import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ConfiguracionCorrida:
    """Parametros de una corrida de la Fase 0."""

    period: int = 7
    n_series_por_sku: int = 10
    block_size: int = 3
    tolerancia_divergencia: float = TOLERANCIA_DIVERGENCIA_POR_DEFECTO
    max_reintentos: int = 20
    semilla_aleatoria: int = 42
    nombre_artefacto: str = NOMBRE_ARTEFACTO_POR_DEFECTO
    minimo_filas: int = MINIMO_FILAS_POR_DEFECTO
    incluir_semilla_en_panel: bool = True


@dataclass(frozen=True, slots=True)
class ResultadoFase0:
    """Salida de una corrida completa de la Fase 0."""

    artefacto: ArtefactoExportado
    bitacora: BitacoraCorrida
    bitacora_path: Path


def _cargar_semilla(ruta: str | Path) -> pd.DataFrame:
    origen = Path(ruta).expanduser()
    if not origen.is_file():
        raise FileNotFoundError(f"no existe la semilla: {origen}")
    frame = pd.read_csv(origen)
    faltan = [c for c in OUTPUT_COLUMNS if c not in frame.columns]
    if faltan:
        raise ValueError(
            f"la semilla debe traer las columnas {list(OUTPUT_COLUMNS)}; "
            f"faltan {faltan}"
        )
    frame = frame.loc[:, list(OUTPUT_COLUMNS)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values(["sku_id", "timestamp"]).reset_index(drop=True)


def _panel_de_sku(
    sku: str,
    grupo: pd.DataFrame,
    config: ConfiguracionCorrida,
) -> tuple[list[pd.DataFrame], int, int]:
    """Devuelve (paneles sinteticos, intentos_bootstrap, rechazos)."""
    demanda = grupo["demand_qty"].to_numpy(dtype="float64")
    lead_time = grupo["lead_time_days"].to_numpy(dtype="float64")
    inicio = pd.Timestamp(grupo["timestamp"].min()).normalize()

    resultado = generar_series_aceptadas(
        demanda,
        period=config.period,
        n_series=config.n_series_por_sku,
        block_size=config.block_size,
        tolerancia=config.tolerancia_divergencia,
        max_reintentos=config.max_reintentos,
        semilla_aleatoria=config.semilla_aleatoria,
    )

    paneles: list[pd.DataFrame] = []
    for indice, serie in enumerate(resultado.series):
        longitud = len(serie)
        paneles.append(
            pd.DataFrame(
                {
                    "sku_id": f"{sku}::syn{indice:03d}",
                    "timestamp": pd.date_range(inicio, periods=longitud, freq="D"),
                    "demand_qty": np.asarray(serie, dtype="float64"),
                    "lead_time_days": np.resize(lead_time, longitud),
                }
            )
        )
    return paneles, resultado.intentos, resultado.rechazos


def ejecutar_fase_0(
    ruta_semilla: str | Path,
    config: ConfiguracionCorrida | None = None,
    *,
    data_root: str | Path | None = None,
) -> ResultadoFase0:
    """Ejecuta la Fase 0 de extremo a extremo y devuelve el artefacto + bitacora."""
    config = config or ConfiguracionCorrida()
    semilla = _cargar_semilla(ruta_semilla)
    _logger.info(
        "Fase 0 iniciada: semilla=%s skus=%s semilla_aleatoria=%s",
        ruta_semilla,
        semilla["sku_id"].nunique(),
        config.semilla_aleatoria,
    )

    limites_globales = limites_lead_time_desde_semilla(
        semilla["lead_time_days"].to_numpy(dtype="float64")
    )

    piezas: list[pd.DataFrame] = []
    if config.incluir_semilla_en_panel:
        piezas.append(semilla.copy())

    skus_procesados = 0
    skus_omitidos = 0
    intentos_bootstrap = 0
    rechazos_bootstrap = 0

    for sku, grupo in semilla.groupby("sku_id", sort=True):
        if len(grupo) < 2 * config.period:
            skus_omitidos += 1
            _logger.warning(
                "SKU %s omitido: %s observaciones (< 2*period=%s)",
                sku,
                len(grupo),
                2 * config.period,
            )
            continue
        paneles, intentos, rechazos = _panel_de_sku(str(sku), grupo, config)
        piezas.extend(paneles)
        intentos_bootstrap += intentos
        rechazos_bootstrap += rechazos
        skus_procesados += 1

    if skus_procesados == 0:
        raise ValueError(
            "ningun SKU de la semilla tiene longitud suficiente para el periodo dado"
        )

    panel_crudo = pd.concat(piezas, ignore_index=True)
    compuerta = aplicar_restricciones_fisicas(
        panel_crudo,
        limites_lead_time=limites_globales,
    )
    reporte = validar_conformidad_o_fallar(compuerta.panel)

    artefacto = exportar_artefacto_csv(
        compuerta.panel,
        config.nombre_artefacto,
        data_root=data_root,
        minimo_filas=config.minimo_filas,
    )

    tasa_rechazo = (
        rechazos_bootstrap / intentos_bootstrap if intentos_bootstrap else 0.0
    )
    bitacora = BitacoraCorrida(
        semilla_ruta=str(ruta_semilla),
        semilla_aleatoria=config.semilla_aleatoria,
        contract_version=CONTRACT_VERSION,
        period=config.period,
        n_series_por_sku=config.n_series_por_sku,
        skus_procesados=skus_procesados,
        skus_omitidos=skus_omitidos,
        tolerancia_divergencia=config.tolerancia_divergencia,
        tasa_rechazo=tasa_rechazo,
        intentos_bootstrap=intentos_bootstrap,
        n_demanda_rectificada=compuerta.n_demanda_rectificada,
        n_lead_time_acotado=compuerta.n_lead_time_acotado,
        row_count=reporte.row_count,
        artefacto_sha256=artefacto.sha256,
        artefacto_path=str(artefacto.path),
    ).cerrar()
    bitacora_path = persistir_bitacora(bitacora, data_root=data_root)

    _logger.info(
        "Fase 0 completada: %s filas, hash %s, tasa de rechazo %.2f%%",
        reporte.row_count,
        artefacto.sha256[:12],
        tasa_rechazo * 100.0,
    )
    return ResultadoFase0(
        artefacto=artefacto,
        bitacora=bitacora,
        bitacora_path=bitacora_path,
    )


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pred-engine-fase0",
        description="Orquestador One-Shot de la Fase 0 de simulacion de pre-ingesta.",
    )
    parser.add_argument("semilla", type=Path, help="Ruta al CSV semilla (Kaggle).")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--period", type=int, default=7)
    parser.add_argument("--n-series", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument(
        "--tolerancia", type=float, default=TOLERANCIA_DIVERGENCIA_POR_DEFECTO
    )
    parser.add_argument("--max-reintentos", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nombre", type=str, default=NOMBRE_ARTEFACTO_POR_DEFECTO)
    parser.add_argument("--minimo-filas", type=int, default=MINIMO_FILAS_POR_DEFECTO)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada CLI. No importa ningun componente del Framework PRED."""
    from pred_engine.comun.logger import configure_json_logger

    configure_json_logger("pred_engine")
    args = _construir_parser().parse_args(argv)
    config = ConfiguracionCorrida(
        period=args.period,
        n_series_por_sku=args.n_series,
        block_size=args.block_size,
        tolerancia_divergencia=args.tolerancia,
        max_reintentos=args.max_reintentos,
        semilla_aleatoria=args.seed,
        nombre_artefacto=args.nombre,
        minimo_filas=args.minimo_filas,
    )
    resultado = ejecutar_fase_0(args.semilla, config, data_root=args.data_root)
    print(
        f"Artefacto: {resultado.artefacto.path} "
        f"({resultado.artefacto.row_count} filas, sha256={resultado.artefacto.sha256})"
    )
    print(f"Bitacora: {resultado.bitacora_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

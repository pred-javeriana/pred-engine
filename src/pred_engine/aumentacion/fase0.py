"""0.4-C1 - Script orquestador de la Fase 0 (One-Shot Execution).

Encadena, en una unica invocacion y sin acoplarse al Framework PRED (Modulo 1):

    semilla -> cuadricula diaria por SKU
            -> remuestreo (MBB + rejection sampling sobre la serie ya acotada
               por las leyes fisicas)
            -> compuerta de restricciones fisicas
            -> validador de conformidad de esquema
            -> exportador CSV bajo politica WORM

La semilla aleatoria se propaga a todas las etapas estocasticas para que la
corrida sea reproducible bit a bit. El metodo de aumento es explicito por
corrida (ADR-016) y queda registrado en la bitacora.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast, get_args

import numpy as np
import pandas as pd

from pred_engine.aumentacion.bitacora import (
    BitacoraCorrida,
    leer_bitacoras,
    persistir_bitacora,
)
from pred_engine.aumentacion.conformidad import validar_conformidad_o_fallar
from pred_engine.aumentacion.contrato import CONTRACT_VERSION, OUTPUT_COLUMNS
from pred_engine.aumentacion.divergencia import (
    TOLERANCIA_DIVERGENCIA_POR_DEFECTO,
)
from pred_engine.aumentacion.errores import (
    DivergenceRejectionExhausted,
    WormOverwriteError,
)
from pred_engine.aumentacion.exportador_csv import (
    MINIMO_FILAS_POR_DEFECTO,
    NOMBRE_ARTEFACTO_POR_DEFECTO,
    ArtefactoExportado,
    exportar_artefacto_csv,
)
from pred_engine.aumentacion.rechazo import (
    GeneradorCandidata,
    generar_series_aceptadas,
    motor_mbb,
    motor_mbb_directo,
)
from pred_engine.aumentacion.restricciones import (
    aplicar_restricciones_fisicas,
    limites_lead_time_desde_semilla,
    truncar_a_unidades_enteras,
)
from pred_engine.aumentacion.rutas import hash_sha256_archivo
from pred_engine.aumentacion.worm import resolver_ruta_artefacto
from pred_engine.comun.logger import get_logger

_logger = get_logger(__name__)

MetodoAumento = Literal["stl-mbb", "mbb-directo"]
METODOS_AUMENTO: tuple[MetodoAumento, ...] = get_args(MetodoAumento)

# Largo de bloque por defecto de cada metodo: 3 dias de residual STL
# (ADR-01-007) o 30 dias de serie completa (seccion 0.2).
BLOQUE_POR_METODO: dict[MetodoAumento, int] = {"stl-mbb": 3, "mbb-directo": 30}

# Con el metodo directo, una candidata pasa la compuerta del 5 % en al menos
# ~10 % de los intentos sobre la semilla Kaggle; 200 reintentos dejan una
# probabilidad de agotamiento por serie del orden de 1e-9.
MAX_REINTENTOS_POR_DEFECTO = 200


@dataclass(frozen=True, slots=True)
class ConfiguracionCorrida:
    """Parametros de una corrida de la Fase 0."""

    period: int = 7
    n_series_por_sku: int = 10
    block_size: int | None = None
    tolerancia_divergencia: float = TOLERANCIA_DIVERGENCIA_POR_DEFECTO
    max_reintentos: int = MAX_REINTENTOS_POR_DEFECTO
    semilla_aleatoria: int = 42
    nombre_artefacto: str = NOMBRE_ARTEFACTO_POR_DEFECTO
    minimo_filas: int = MINIMO_FILAS_POR_DEFECTO
    incluir_semilla_en_panel: bool = True
    metodo: MetodoAumento = "stl-mbb"
    ruido_relativo: float = 0.03
    # Pares (columna canonica, columna de la semilla) para semillas que no
    # traen las cabeceras canonicas, p. ej. ("sku_id", "Item_ID").
    mapeo_columnas: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.metodo not in METODOS_AUMENTO:
            raise ValueError(
                f"metodo de aumento desconocido {self.metodo!r}; "
                f"use uno de {list(METODOS_AUMENTO)}"
            )
        for nombre in ("period", "n_series_por_sku", "max_reintentos"):
            if getattr(self, nombre) < 1:
                raise ValueError(f"{nombre} debe ser un entero positivo")
        if self.block_size is not None and self.block_size < 1:
            raise ValueError("block_size debe ser un entero positivo")
        if not 0 <= self.ruido_relativo < 1:
            raise ValueError("ruido_relativo debe estar en [0, 1)")
        canonicas = [canonica for canonica, _ in self.mapeo_columnas]
        desconocidas = sorted(set(canonicas) - set(OUTPUT_COLUMNS))
        if desconocidas:
            raise ValueError(
                f"columnas canonicas desconocidas en el mapeo: {desconocidas}"
            )
        if len(set(canonicas)) != len(canonicas):
            raise ValueError("el mapeo repite una columna canonica")

    @property
    def bloque_efectivo(self) -> int:
        if self.block_size is not None:
            return self.block_size
        return BLOQUE_POR_METODO[self.metodo]


@dataclass(frozen=True, slots=True)
class ResultadoFase0:
    """Salida de una corrida completa de la Fase 0."""

    artefacto: ArtefactoExportado
    bitacora: BitacoraCorrida
    bitacora_path: Path
    # True cuando el artefacto WORM ya existia y lo produjo la misma corrida.
    reutilizada: bool = False


def _cargar_semilla(
    ruta: str | Path, mapeo: tuple[tuple[str, str], ...] = ()
) -> pd.DataFrame:
    origen = Path(ruta).expanduser()
    if not origen.is_file():
        raise FileNotFoundError(f"no existe la semilla: {origen}")
    frame = pd.read_csv(origen)
    renombre = {fuente: canonica for canonica, fuente in mapeo}
    ausentes = sorted(fuente for fuente in renombre if fuente not in frame.columns)
    if ausentes:
        raise ValueError(f"la semilla no trae las columnas del mapeo: {ausentes}")
    frame = frame.rename(columns=renombre)
    faltan = [c for c in OUTPUT_COLUMNS if c not in frame.columns]
    if faltan:
        raise ValueError(
            f"la semilla debe traer las columnas {list(OUTPUT_COLUMNS)} "
            f"(o mapearlas con --columna); faltan {faltan}"
        )
    frame = frame.loc[:, list(OUTPUT_COLUMNS)].copy()
    frame["sku_id"] = frame["sku_id"].astype(str).str.strip()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.normalize()
    for columna in ("demand_qty", "lead_time_days"):
        frame[columna] = pd.to_numeric(frame[columna], errors="raise")
    return frame.sort_values(["sku_id", "timestamp"]).reset_index(drop=True)


def cuadricula_diaria(semilla: pd.DataFrame) -> pd.DataFrame:
    """Lleva cada SKU de la semilla a un calendario diario continuo.

    Las transacciones del mismo dia se suman; los dias sin registro tienen
    demanda 0 y heredan el lead time vigente (hacia adelante y, al inicio,
    hacia atras). Sin este paso una semilla transaccional (fechas salteadas)
    se comprimiria en dias consecutivos y perderia su intermitencia.
    """
    piezas: list[pd.DataFrame] = []
    for sku, grupo in semilla.groupby("sku_id", sort=True):
        diario = grupo.groupby("timestamp").agg(
            demand_qty=("demand_qty", "sum"),
            lead_time_days=("lead_time_days", "max"),
        )
        calendario = pd.date_range(diario.index.min(), diario.index.max(), freq="D")
        diario = cast(pd.DataFrame, diario.reindex(calendario))
        demanda = cast(pd.Series, diario["demand_qty"]).fillna(0.0)
        lead_time = cast(pd.Series, diario["lead_time_days"]).ffill().bfill()
        piezas.append(
            pd.DataFrame(
                {
                    "sku_id": sku,
                    "timestamp": calendario,
                    "demand_qty": demanda.to_numpy(dtype="float64"),
                    "lead_time_days": lead_time.to_numpy(dtype="float64"),
                }
            )
        )
    return pd.concat(piezas, ignore_index=True)


def _semilla_de_sku(semilla_aleatoria: int, sku: str) -> int:
    """Semilla propia por SKU: SKUs distintos no comparten el mismo remuestreo."""
    digesto = hashlib.blake2b(
        f"{semilla_aleatoria}:{sku}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digesto, "big") % (2**32)


def _leyes_fisicas(candidata: np.ndarray) -> np.ndarray:
    """0.3-A1/A2 antes de la compuerta: demanda no negativa y entera."""
    return truncar_a_unidades_enteras(np.maximum(candidata, 0.0)).astype("float64")


def _motor(config: ConfiguracionCorrida, demanda: np.ndarray) -> GeneradorCandidata:
    if config.metodo == "mbb-directo":
        return motor_mbb_directo(
            demanda,
            block_size=config.bloque_efectivo,
            ruido_relativo=config.ruido_relativo,
        )
    return motor_mbb(demanda, period=config.period, block_size=config.bloque_efectivo)


def _longitud_minima(config: ConfiguracionCorrida) -> int:
    if config.metodo == "mbb-directo":
        return config.bloque_efectivo
    return max(2 * config.period, config.bloque_efectivo)


def _panel_de_sku(
    sku: str,
    grupo: pd.DataFrame,
    config: ConfiguracionCorrida,
) -> tuple[list[pd.DataFrame], int, int, int]:
    """Devuelve (paneles sinteticos, intentos, rechazos, demandas rectificadas)."""
    demanda = grupo["demand_qty"].to_numpy(dtype="float64")
    lead_time = grupo["lead_time_days"].to_numpy(dtype="float64")
    calendario = pd.DatetimeIndex(grupo["timestamp"])

    resultado = generar_series_aceptadas(
        demanda,
        period=config.period,
        n_series=config.n_series_por_sku,
        block_size=config.bloque_efectivo,
        tolerancia=config.tolerancia_divergencia,
        max_reintentos=config.max_reintentos,
        semilla_aleatoria=_semilla_de_sku(config.semilla_aleatoria, sku),
        generador=_motor(config, demanda),
        postproceso=_leyes_fisicas,
    )
    rectificadas = sum(int(np.count_nonzero(c < 0)) for c in resultado.crudas)

    # Cada serie sintetica comparte el calendario y el lead time de su SKU
    # semilla, asi que todas terminan en la misma fecha que la historia real.
    paneles = [
        pd.DataFrame(
            {
                "sku_id": f"{sku}::syn{indice:03d}",
                "timestamp": calendario,
                "demand_qty": serie,
                "lead_time_days": lead_time,
            }
        )
        for indice, serie in enumerate(resultado.series)
    ]
    return paneles, resultado.intentos, resultado.rechazos, rectificadas


def huella_corrida(config: ConfiguracionCorrida, semilla_sha256: str) -> str:
    """Identidad estable de una corrida: semilla, contrato y configuracion."""
    contenido = {
        "contract_version": CONTRACT_VERSION,
        "semilla_sha256": semilla_sha256,
        "configuracion": _configuracion_serializable(config),
    }
    texto = json.dumps(contenido, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(texto.encode()).hexdigest()


def _configuracion_serializable(config: ConfiguracionCorrida) -> dict[str, object]:
    datos = asdict(config)
    datos["mapeo_columnas"] = dict(config.mapeo_columnas)
    datos["block_size"] = config.bloque_efectivo
    return datos


def _corrida_previa(
    destino: Path, huella: str, *, data_root: str | Path | None
) -> ResultadoFase0 | None:
    """Bitacora de la corrida identica que ya deposito ``destino``, si existe."""
    sha_actual = hash_sha256_archivo(destino)
    campos = set(BitacoraCorrida.__dataclass_fields__)
    for datos in leer_bitacoras(data_root=data_root):
        if datos.get("huella_corrida") != huella:
            continue
        if Path(str(datos.get("artefacto_path"))).resolve() != destino.resolve():
            continue
        if datos.get("artefacto_sha256") != sha_actual:
            continue
        bitacora = BitacoraCorrida(**{k: v for k, v in datos.items() if k in campos})
        return ResultadoFase0(
            artefacto=ArtefactoExportado(
                path=destino, sha256=sha_actual, row_count=bitacora.row_count
            ),
            bitacora=bitacora,
            bitacora_path=Path(datos["_ruta"]),
            reutilizada=True,
        )
    return None


def ejecutar_fase_0(
    ruta_semilla: str | Path,
    config: ConfiguracionCorrida | None = None,
    *,
    data_root: str | Path | None = None,
    reutilizar: bool = False,
) -> ResultadoFase0:
    """Ejecuta la Fase 0 de extremo a extremo y devuelve el artefacto + bitacora.

    Con ``reutilizar=True`` una segunda invocacion identica (misma semilla y
    configuracion) devuelve el artefacto WORM ya depositado en lugar de fallar;
    cualquier otra corrida sobre el mismo destino sigue siendo rechazada.
    """
    config = config or ConfiguracionCorrida()
    iniciada_en = datetime.now(UTC).isoformat()
    semilla_cruda = _cargar_semilla(ruta_semilla, config.mapeo_columnas)
    semilla_sha = hash_sha256_archivo(Path(ruta_semilla).expanduser())
    huella = huella_corrida(config, semilla_sha)

    destino = resolver_ruta_artefacto(config.nombre_artefacto, data_root=data_root)
    if reutilizar and destino.exists():
        previa = _corrida_previa(destino, huella, data_root=data_root)
        if previa is not None:
            _logger.info("Fase 0 reutilizada: %s (huella %s)", destino, huella[:12])
            return previa

    semilla = cuadricula_diaria(semilla_cruda)
    _logger.info(
        "Fase 0 iniciada: semilla=%s skus=%s metodo=%s bloque=%s semilla_aleatoria=%s",
        ruta_semilla,
        semilla["sku_id"].nunique(),
        config.metodo,
        config.bloque_efectivo,
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
    rectificadas_bootstrap = 0
    minimo = _longitud_minima(config)

    for sku, grupo in semilla.groupby("sku_id", sort=True):
        if len(grupo) < minimo:
            skus_omitidos += 1
            _logger.warning(
                "SKU %s omitido: %s dias (< %s requeridos por %s)",
                sku,
                len(grupo),
                minimo,
                config.metodo,
            )
            continue
        paneles, intentos, rechazos, rectificadas = _panel_de_sku(
            str(sku), grupo, config
        )
        piezas.extend(paneles)
        intentos_bootstrap += intentos
        rechazos_bootstrap += rechazos
        rectificadas_bootstrap += rectificadas
        skus_procesados += 1

    if skus_procesados == 0:
        raise ValueError(
            "ningun SKU de la semilla tiene longitud suficiente para el metodo "
            f"{config.metodo} (minimo {minimo} dias)"
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
        n_demanda_rectificada=compuerta.n_demanda_rectificada + rectificadas_bootstrap,
        n_lead_time_acotado=compuerta.n_lead_time_acotado,
        row_count=reporte.row_count,
        artefacto_sha256=artefacto.sha256,
        artefacto_path=str(artefacto.path),
        iniciada_en=iniciada_en,
        metodo=config.metodo,
        block_size=config.bloque_efectivo,
        ruido_relativo=config.ruido_relativo if config.metodo == "mbb-directo" else 0.0,
        max_reintentos=config.max_reintentos,
        mapeo_columnas=dict(config.mapeo_columnas),
        semilla_sha256=semilla_sha,
        huella_corrida=huella,
        configuracion=_configuracion_serializable(config),
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


def parsear_columna(texto: str) -> tuple[str, str]:
    """``canonica=origen`` -> ``(canonica, origen)``; usado por ambos CLIs."""
    canonica, separador, origen = texto.partition("=")
    if not separador or not canonica.strip() or not origen.strip():
        raise argparse.ArgumentTypeError(
            f"use CANONICA=ORIGEN (p. ej. sku_id=Item_ID); se recibio {texto!r}"
        )
    return canonica.strip(), origen.strip()


def agregar_opciones_fase0(
    parser: argparse._ActionsContainer, prefijo: str = ""
) -> None:
    """Opciones de la corrida, compartidas por este CLI y ``pred-engine run``.

    ``prefijo`` (p. ej. ``"m0-"``) evita choques de nombres en otro CLI; la
    configuracion se lee con el mismo prefijo en `configuracion_desde_argumentos`.
    """
    parser.add_argument(
        f"--{prefijo}metodo",
        choices=METODOS_AUMENTO,
        default="stl-mbb",
        help=(
            "stl-mbb: STL + MBB de residuales (ADR-01-007). mbb-directo: MBB de la "
            "serie diaria que conserva las rachas de cero (semillas intermitentes)"
        ),
    )
    parser.add_argument(
        f"--{prefijo}columna",
        type=parsear_columna,
        action="append",
        default=[],
        metavar="CANONICA=ORIGEN",
        help="Mapea una columna de la semilla a la canonica (repetible)",
    )
    parser.add_argument(f"--{prefijo}period", type=int, default=7)
    parser.add_argument(f"--{prefijo}n-series", type=int, default=10)
    parser.add_argument(
        f"--{prefijo}block-size",
        type=int,
        default=None,
        help="Largo de bloque; por defecto 3 (stl-mbb) o 30 (mbb-directo)",
    )
    parser.add_argument(
        f"--{prefijo}ruido",
        type=float,
        default=0.03,
        help="mbb-directo: desviacion del ruido relativa a la media positiva",
    )
    parser.add_argument(
        f"--{prefijo}tolerancia", type=float, default=TOLERANCIA_DIVERGENCIA_POR_DEFECTO
    )
    parser.add_argument(
        f"--{prefijo}max-reintentos", type=int, default=MAX_REINTENTOS_POR_DEFECTO
    )
    parser.add_argument(f"--{prefijo}seed", type=int, default=42)
    parser.add_argument(
        f"--{prefijo}nombre", type=str, default=NOMBRE_ARTEFACTO_POR_DEFECTO
    )
    parser.add_argument(
        f"--{prefijo}minimo-filas", type=int, default=MINIMO_FILAS_POR_DEFECTO
    )


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pred-engine-fase0",
        description="Orquestador One-Shot de la Fase 0 de simulacion de pre-ingesta.",
    )
    parser.add_argument("semilla", type=Path, help="Ruta al CSV semilla (Kaggle).")
    parser.add_argument("--data-root", type=Path, default=None)
    agregar_opciones_fase0(parser)
    parser.add_argument(
        "--reutilizar",
        action="store_true",
        help="Si el artefacto ya existe y lo produjo esta misma corrida, reutilizarlo",
    )
    return parser


def configuracion_desde_argumentos(
    args: argparse.Namespace, prefijo: str = ""
) -> ConfiguracionCorrida:
    destino = prefijo.replace("-", "_")

    def valor(nombre: str) -> Any:
        return getattr(args, destino + nombre)

    return ConfiguracionCorrida(
        period=valor("period"),
        n_series_por_sku=valor("n_series"),
        block_size=valor("block_size"),
        tolerancia_divergencia=valor("tolerancia"),
        max_reintentos=valor("max_reintentos"),
        semilla_aleatoria=valor("seed"),
        nombre_artefacto=valor("nombre"),
        minimo_filas=valor("minimo_filas"),
        metodo=valor("metodo"),
        ruido_relativo=valor("ruido"),
        mapeo_columnas=tuple(valor("columna")),
    )


# Errores esperables de una corrida: se informan sin traza al operador.
ERRORES_FASE0: tuple[type[Exception], ...] = (
    FileNotFoundError,
    ValueError,
    DivergenceRejectionExhausted,
    WormOverwriteError,
)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada CLI. No importa ningun componente del Framework PRED."""
    from pred_engine.comun.logger import configure_json_logger

    configure_json_logger("pred_engine")
    args = _construir_parser().parse_args(argv)
    try:
        resultado = ejecutar_fase_0(
            args.semilla,
            configuracion_desde_argumentos(args),
            data_root=args.data_root,
            reutilizar=args.reutilizar,
        )
    except ERRORES_FASE0 as exc:
        _logger.error("Fase 0 fallida: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Artefacto: {resultado.artefacto.path} "
        f"({resultado.artefacto.row_count} filas, sha256={resultado.artefacto.sha256})"
    )
    print(f"Bitacora: {resultado.bitacora_path}")
    if resultado.reutilizada:
        print("Reutilizada: el artefacto ya existia para esta misma corrida")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

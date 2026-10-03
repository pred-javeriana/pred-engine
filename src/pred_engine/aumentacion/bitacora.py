"""0.4-C2 - Bitacora estructurada de la corrida de handoff.

Registra de forma reconstruible los parametros de una corrida de la Fase 0.
La bitacora se persiste junto a la corrida (``{data_root}/logs/``), nunca
dentro del directorio crudo inmutable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pred_engine.aumentacion.rutas import resolver_rutas
from pred_engine.comun.logger import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BitacoraCorrida:
    """Parametros y resultados auditables de una corrida de la Fase 0."""

    semilla_ruta: str
    semilla_aleatoria: int
    contract_version: str
    period: int
    n_series_por_sku: int
    skus_procesados: int
    skus_omitidos: int
    tolerancia_divergencia: float
    tasa_rechazo: float
    intentos_bootstrap: int
    n_demanda_rectificada: int
    n_lead_time_acotado: int
    row_count: int
    artefacto_sha256: str
    artefacto_path: str
    iniciada_en: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    finalizada_en: str | None = None
    # Metodo de aumento (ADR-016) y parametros efectivos de la corrida.
    metodo: str = "stl-mbb"
    block_size: int = 3
    ruido_relativo: float = 0.0
    max_reintentos: int = 20
    mapeo_columnas: dict[str, str] = field(default_factory=dict)
    semilla_sha256: str | None = None
    # Identidad de la corrida: semilla + configuracion. Permite reconocer un
    # artefacto WORM ya producido por exactamente la misma corrida.
    huella_corrida: str | None = None
    configuracion: dict[str, Any] = field(default_factory=dict)

    def cerrar(self) -> BitacoraCorrida:
        datos = asdict(self)
        datos["finalizada_en"] = datetime.now(UTC).isoformat()
        return BitacoraCorrida(**datos)


def leer_bitacoras(*, data_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Bitacoras JSON de ``{data_root}/logs/``, de la mas reciente a la mas antigua."""
    rutas = resolver_rutas(data_root)
    bitacoras: list[dict[str, Any]] = []
    for ruta in sorted(rutas.logs.glob("fase0_*.json"), reverse=True):
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _logger.warning("Bitacora ilegible ignorada: %s", ruta)
            continue
        if isinstance(datos, dict):
            datos["_ruta"] = str(ruta)
            bitacoras.append(datos)
    return bitacoras


def persistir_bitacora(
    bitacora: BitacoraCorrida,
    *,
    data_root: str | Path | None = None,
) -> Path:
    """Escribe la bitacora como JSON fuera del directorio crudo."""
    rutas = resolver_rutas(data_root)
    marca = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    destino = rutas.logs / f"fase0_{marca}_seed{bitacora.semilla_aleatoria}.json"
    destino.write_text(
        json.dumps(asdict(bitacora), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _logger.info("Bitacora de la corrida persistida en %s", destino)
    return destino

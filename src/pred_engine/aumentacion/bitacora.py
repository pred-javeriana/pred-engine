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

    def cerrar(self) -> BitacoraCorrida:
        datos = asdict(self)
        datos["finalizada_en"] = datetime.now(UTC).isoformat()
        return BitacoraCorrida(**datos)


def persistir_bitacora(
    bitacora: BitacoraCorrida,
    *,
    data_root: str | Path | None = None,
) -> Path:
    """Escribe la bitacora como JSON fuera del directorio crudo."""
    rutas = resolver_rutas(data_root)
    marca = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    destino = rutas.logs / f"fase0_{marca}_seed{bitacora.semilla_aleatoria}.json"
    destino.write_text(
        json.dumps(asdict(bitacora), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _logger.info("Bitacora de la corrida persistida en %s", destino)
    return destino

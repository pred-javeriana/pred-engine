"""Huella determinista de la configuracion experimental de una corrida."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pred_engine.comun.logger import get_logger
from pred_engine.optimizacion.control_reanudacion.contratos import (
    ManifiestoCorrida,
    SolicitudCorrida,
)
from pred_engine.optimizacion.control_reanudacion.errores import (
    IncompatibilidadCorridaError,
)

_logger = get_logger(__name__)

_CAMPOS_HUELLA: tuple[str, ...] = (
    "familia",
    "sku_id",
    "seed",
    "metrica_objetivo",
    "validacion",
    "espacio_busqueda",
    "optimizador",
    "n_observaciones",
    "huella_serie",
    "backend",
)


def _canonico(valor: Any) -> Any:
    if hasattr(valor, "model_dump"):
        return valor.model_dump(mode="python")
    if isinstance(valor, Mapping):
        return {str(k): _canonico(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_canonico(v) for v in valor]
    return valor


def payload_huella(solicitud: SolicitudCorrida) -> dict[str, Any]:
    datos = solicitud.model_dump(mode="python")
    return {campo: _canonico(datos[campo]) for campo in _CAMPOS_HUELLA}


def calcular_huella(solicitud: SolicitudCorrida) -> str:
    payload = payload_huella(solicitud)
    texto = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def motivos_incompatibilidad(
    solicitud: SolicitudCorrida, manifiesto: ManifiestoCorrida
) -> tuple[str, ...]:
    """Lista explicita de validaciones que impiden reanudar."""
    motivos: list[str] = []
    esperada = calcular_huella(solicitud)
    if manifiesto.familia != solicitud.familia:
        motivos.append("familia")
    if manifiesto.sku_id != solicitud.sku_id:
        motivos.append("sku_id")
    if manifiesto.seed != solicitud.seed:
        motivos.append("seed")
    if manifiesto.metrica_objetivo != solicitud.metrica_objetivo:
        motivos.append("metrica_objetivo")
    if manifiesto.configuracion_validacion != solicitud.validacion:
        motivos.append("validacion")
    if manifiesto.n_trials_objetivo != solicitud.optimizador.n_trials_objetivo:
        motivos.append("n_trials_objetivo")
    if manifiesto.backend != solicitud.backend:
        motivos.append("backend")
    if manifiesto.fingerprint_configuracion != esperada:
        # Cubre espacio, reglas de poda, serie y cualquier campo no
        # comparado arriba. Siempre se informa si la huella global diverge.
        if "huella" not in motivos:
            motivos.append("huella")
    return tuple(motivos)


def verificar_compatibilidad(
    solicitud: SolicitudCorrida, manifiesto: ManifiestoCorrida
) -> None:
    motivos = motivos_incompatibilidad(solicitud, manifiesto)
    if not motivos:
        return
    _logger.error(
        "Incompatibilidad corrida=%s motivos=%s",
        solicitud.run_id,
        ",".join(motivos),
    )
    raise IncompatibilidadCorridaError(motivos)

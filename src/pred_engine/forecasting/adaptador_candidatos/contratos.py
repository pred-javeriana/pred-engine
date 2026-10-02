"""Salida del adaptador: candidatos validos y fallos aislados por candidato."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pred_engine.comun.modelos.manifiesto_candidatos import (
    Candidato,
    ContextoParticion,
)

# Rechazos de lote (ADR-03-005): comprometen la comparacion de TODOS los
# candidatos, asi que la reserva no se abre.
MotivoRechazo = Literal[
    "manifiesto_ilegible",
    "version_no_soportada",
    "contexto_incompatible",
    "handoff_posterior_a_reserva",
    "candidato_id_duplicado",
    "sku_fuera_del_panel",
    "error_no_clasificado",
]

# Fallos de candidato: solo afectan a ese candidato; la corrida continua.
CodigoFallo = Literal[
    "modelo_no_registrado",
    "campo_faltante",
    "campo_sobrante",
    "tipo_incorrecto",
    "fuera_de_dominio",
    "semilla_ausente",
    "pesos_no_disponibles",
]


@dataclass(frozen=True, slots=True)
class FalloCandidato:
    """Registro de un candidato que no puede evaluarse; nunca un descarte mudo."""

    sku: str
    candidato_id: str
    modelo: str | None
    campo: str
    codigo_error: CodigoFallo
    mensaje: str


@dataclass(frozen=True, slots=True)
class HandoffValidado:
    """Ganadores por familia de M2 que validaron; M3 elige el campeon entre ellos."""

    contexto: ContextoParticion
    run_id_m2: str
    candidatos: tuple[Candidato, ...]
    fallos: tuple[FalloCandidato, ...]

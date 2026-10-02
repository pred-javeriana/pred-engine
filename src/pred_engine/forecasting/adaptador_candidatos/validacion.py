"""Validacion del handoff M2 -> M3 en dos niveles (ADR-03-005).

Una sola pregunta clasifica cada error: ¿compromete la comparacion de todos
los candidatos o solo la de uno? Lo primero rechaza el lote antes de abrir la
reserva (fail-closed); lo segundo produce un `FalloCandidato` y la corrida
sigue con los demas.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import datetime
from typing import Any

from pydantic import TypeAdapter, ValidationError

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos.manifiesto_candidatos import (
    VERSION_ESQUEMA_MANIFIESTO,
    Candidato,
    ContextoParticion,
    ManifiestoCandidatos,
)
from pred_engine.comun.modelos.modelos_fundacionales.configuracion import (
    CHRONOS2_ZERO_SHOT,
    ConfiguracionFundacional,
)
from pred_engine.comun.modelos.modelos_fundacionales.errores import (
    ModeloFundacionalNoDisponibleError,
)
from pred_engine.comun.modelos.modelos_fundacionales.pipeline import cargar_pipeline
from pred_engine.comun.walkforward.protocolos import FabricaPronosticador
from pred_engine.forecasting.adaptador_candidatos.contratos import (
    CodigoFallo,
    FalloCandidato,
    HandoffValidado,
    MotivoRechazo,
)
from pred_engine.forecasting.adaptador_candidatos.errores import LoteRechazadoError
from pred_engine.forecasting.adaptador_candidatos.registro import FABRICAS

_logger = get_logger(__name__)

_ADAPTADOR_CANDIDATO: TypeAdapter[Candidato] = TypeAdapter(Candidato)


def validar_manifiesto(
    contenido: str | bytes,
    *,
    esperado: ContextoParticion,
    skus_panel: Collection[str],
    reserva_abierta_en: datetime | None = None,
    fabricas: Mapping[str, FabricaPronosticador] = FABRICAS,
    cargar_fundacional: Callable[[ConfiguracionFundacional], object] = (
        cargar_pipeline
    ),
) -> HandoffValidado:
    """Valida el lote y luego cada candidato; nunca descarta uno sin registro."""
    try:
        manifiesto = _validar_lote(
            contenido,
            esperado=esperado,
            skus_panel=skus_panel,
            reserva_abierta_en=reserva_abierta_en,
        )
        validos, fallos = _validar_candidatos(manifiesto.candidatos, fabricas)
        validos, fallos_pesos = _verificar_pesos(validos, cargar_fundacional)
    except LoteRechazadoError:
        raise
    except Exception as exc:
        # Lo no clasificado se cierra: nunca degrada a un fallo local.
        raise _rechazar("error_no_clasificado", f"{type(exc).__name__}: {exc}") from exc

    _logger.info(
        "Handoff validado run_id_m2=%s candidatos=%d fallos=%d",
        manifiesto.run_id_m2,
        len(validos),
        len(fallos) + len(fallos_pesos),
    )
    return HandoffValidado(
        contexto=manifiesto.contexto,
        run_id_m2=manifiesto.run_id_m2,
        candidatos=tuple(validos),
        fallos=tuple(fallos + fallos_pesos),
    )


def _validar_lote(
    contenido: str | bytes,
    *,
    esperado: ContextoParticion,
    skus_panel: Collection[str],
    reserva_abierta_en: datetime | None,
) -> ManifiestoCandidatos:
    try:
        datos = json.loads(contenido)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _rechazar("manifiesto_ilegible", f"JSON invalido: {exc}") from exc
    if not isinstance(datos, dict):
        raise _rechazar("manifiesto_ilegible", "el manifiesto no es un objeto JSON")
    version = datos.get("schema_version")
    if version != VERSION_ESQUEMA_MANIFIESTO:
        raise _rechazar(
            "version_no_soportada",
            f"schema_version={version!r}, se soporta {VERSION_ESQUEMA_MANIFIESTO}",
        )
    try:
        manifiesto = ManifiestoCandidatos.model_validate_json(contenido)
    except ValidationError as exc:
        raise _rechazar("manifiesto_ilegible", _resumen(exc)) from exc

    if manifiesto.contexto != esperado:
        distintos = sorted(
            campo
            for campo, valor in esperado.model_dump().items()
            if manifiesto.contexto.model_dump()[campo] != valor
        )
        raise _rechazar(
            "contexto_incompatible",
            f"difiere de la particion de 3.1 en {distintos}",
        )
    if reserva_abierta_en is not None and manifiesto.emitido_en >= reserva_abierta_en:
        raise _rechazar(
            "handoff_posterior_a_reserva",
            f"emitido_en={manifiesto.emitido_en.isoformat()} no es anterior a la "
            f"apertura de la reserva {reserva_abierta_en.isoformat()}",
        )

    ids = [_identidad(crudo, "candidato_id") for crudo in manifiesto.candidatos]
    duplicados = sorted(i for i, n in Counter(ids).items() if n > 1)
    if duplicados:
        raise _rechazar("candidato_id_duplicado", f"repetidos: {duplicados}")
    panel = set(skus_panel)
    ajenos = sorted(
        {_identidad(crudo, "sku") for crudo in manifiesto.candidatos} - panel
    )
    if ajenos:
        raise _rechazar("sku_fuera_del_panel", f"no existen en el panel M1: {ajenos}")
    return manifiesto


def _identidad(crudo: Mapping[str, Any], campo: str) -> str:
    # Sin identidad no hay a quien atribuir el fallo: el lote es ilegible.
    valor = crudo.get(campo)
    if not isinstance(valor, str) or not valor.strip():
        raise _rechazar("manifiesto_ilegible", f"candidato sin {campo} legible")
    return valor.strip()


def _validar_candidatos(
    crudos: Sequence[Mapping[str, Any]],
    fabricas: Mapping[str, FabricaPronosticador],
) -> tuple[list[Candidato], list[FalloCandidato]]:
    validos: list[Candidato] = []
    fallos: list[FalloCandidato] = []
    for crudo in crudos:
        try:
            candidato = _ADAPTADOR_CANDIDATO.validate_python(crudo)
        except ValidationError as exc:
            campo, codigo = _clasificar(exc)
            fallos.append(_fallo(crudo, campo, codigo, _resumen(exc)))
            continue
        if candidato.familia not in fabricas:
            fallos.append(
                _fallo(
                    crudo,
                    "familia",
                    "modelo_no_registrado",
                    f"la familia {candidato.familia!r} no esta habilitada",
                )
            )
            continue
        validos.append(candidato)
    return validos, fallos


def _verificar_pesos(
    validos: list[Candidato],
    cargar_fundacional: Callable[[ConfiguracionFundacional], object],
) -> tuple[list[Candidato], list[FalloCandidato]]:
    fundacionales = [c for c in validos if c.familia == "foundation"]
    if not fundacionales:
        return validos, []
    try:
        # Una sola carga por corrida: los pesos son los mismos para todo SKU.
        cargar_fundacional(CHRONOS2_ZERO_SHOT)
    except ModeloFundacionalNoDisponibleError as exc:
        fallos = [
            _fallo(
                c.model_dump(),
                "configuracion.revision",
                "pesos_no_disponibles",
                str(exc),
            )
            for c in fundacionales
        ]
        return [c for c in validos if c.familia != "foundation"], fallos
    return validos, []


def _clasificar(exc: ValidationError) -> tuple[str, CodigoFallo]:
    error = exc.errors()[0]
    tipo = error["type"]
    ruta = [str(parte) for parte in error["loc"]]
    # El primer elemento de `loc` es la etiqueta de la union, no un campo.
    campo = ".".join(ruta[1:]) if len(ruta) > 1 else "familia"
    if tipo.startswith("union_tag"):
        return "familia", "modelo_no_registrado"
    if tipo == "literal_error" and campo == "modelo":
        return campo, "modelo_no_registrado"
    if tipo == "missing":
        return campo, "semilla_ausente" if campo == "semilla" else "campo_faltante"
    if tipo == "extra_forbidden":
        return campo, "campo_sobrante"
    if tipo.endswith("_type"):
        return campo, "tipo_incorrecto"
    return campo, "fuera_de_dominio"


def _fallo(
    crudo: Mapping[str, Any], campo: str, codigo: CodigoFallo, detalle: str
) -> FalloCandidato:
    sku = str(crudo.get("sku"))
    candidato_id = str(crudo.get("candidato_id"))
    declarado = crudo.get("modelo")
    modelo = declarado if isinstance(declarado, str) else None
    mensaje = (
        f"sku={sku} candidato_id={candidato_id} modelo={modelo} campo={campo}: "
        f"{detalle}"
    )
    _logger.error(
        "Candidato rechazado sku=%s candidato_id=%s modelo=%s campo=%s codigo=%s",
        sku,
        candidato_id,
        modelo,
        campo,
        codigo,
    )
    return FalloCandidato(
        sku=sku,
        candidato_id=candidato_id,
        modelo=modelo,
        campo=campo,
        codigo_error=codigo,
        mensaje=mensaje,
    )


def _rechazar(motivo: MotivoRechazo, detalle: str) -> LoteRechazadoError:
    _logger.error("Lote rechazado motivo=%s detalle=%s", motivo, detalle)
    return LoteRechazadoError(motivo, detalle)


def _resumen(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or '<raiz>'}: {e['msg']}"
        for e in exc.errors()
    )

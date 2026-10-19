"""Ajuste unico en t* y pronosticos causales por origen reservado (3.3)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from pred_engine.comun.logger import get_logger
from pred_engine.comun.modelos.manifiesto_candidatos import Candidato
from pred_engine.comun.modelos.modelos_fundacionales.pipeline import PipelineFundacional
from pred_engine.comun.reserva import ReserveCut
from pred_engine.forecasting.adaptador_candidatos import (
    FABRICAS,
    AdaptadorCandidato,
    FabricaAdaptador,
    instanciar,
)
from pred_engine.forecasting.evaluaciones.calculo_errores import PronosticoFechado
from pred_engine.forecasting.generador_pronosticos.contratos import (
    CodigoFalloPronostico,
    ConfiguracionPronosticos,
    FalloPronostico,
    ResultadoPronosticos,
    UnidadPronostico,
    VentanaReserva,
)
from pred_engine.forecasting.generador_pronosticos.particion import (
    separar_serie,
    ventanas_reserva,
)

_logger = get_logger(__name__)


def generar_pronosticos(
    candidato: Candidato,
    serie: pd.Series,
    corte: ReserveCut,
    configuracion: ConfiguracionPronosticos,
    *,
    fabricas: Mapping[str, FabricaAdaptador] = FABRICAS,
    pipeline_fundacional: PipelineFundacional | None = None,
    confirmadas: Sequence[UnidadPronostico] = (),
    confirmar: Callable[[UnidadPronostico], None] | None = None,
) -> ResultadoPronosticos:
    """Genera todas las ventanas completas; nunca ejecuta HPO ni retunea.

    Cada origen recibe solo las observaciones reservadas hasta ese origen.
    Se ajusta una vez con historia <= t*, tambien al reanudar un candidato
    parcialmente confirmado. Sus pronosticos ya confirmados no se recalculan.

    ``confirmar`` recibe cada unidad nueva, valida o fallida. El coordinador
    debe guardar resultado y marcador atomicamente antes de devolver control.
    Los errores de esa escritura se propagan, no se confunden con una falla
    del modelo. Un reinicio pasa las unidades durables en ``confirmadas``.
    """
    ventanas = ventanas_reserva(corte, configuracion.horizonte)
    restauradas = _restaurar(candidato, configuracion, ventanas, confirmadas)
    if len(restauradas) == len(ventanas) and ventanas:
        return ResultadoPronosticos(
            candidato, tuple(restauradas[v.origen] for v in ventanas)
        )
    historia, reserva = separar_serie(candidato.sku, serie, corte, configuracion)
    modelo: AdaptadorCandidato | None = None
    error_ajuste: Exception | None = None
    try:
        modelo = instanciar(
            candidato, fabricas=fabricas, pipeline_fundacional=pipeline_fundacional
        )
        modelo.fit(historia)
    except Exception as exc:
        error_ajuste = exc

    unidades: list[UnidadPronostico] = []
    for ventana in ventanas:
        if ventana.origen in restauradas:
            unidades.append(restauradas[ventana.origen])
            continue
        fallo: FalloPronostico | None = None
        valores = np.full(configuracion.horizonte, np.nan)
        if error_ajuste is not None:
            fallo = _fallo(candidato, ventana, "fit", "ajuste_fallido", error_ajuste)
        else:
            assert modelo is not None
            valores, fallo = _pronosticar(candidato, modelo, reserva, ventana)
        valores.setflags(write=False)
        unidad = UnidadPronostico(
            candidato=candidato,
            codigo_version=configuracion.codigo_version,
            emitido_en=datetime.now(UTC),
            pronostico=PronosticoFechado(ventana.origen, ventana.fechas, valores),
            fallo=fallo,
        )
        # Fuera de los manejadores del modelo: un fallo de commit detiene la
        # ejecucion para que solo se retome desde el ultimo marcador durable.
        if confirmar is not None:
            confirmar(unidad)
        unidades.append(unidad)
    _logger.info(
        "Pronosticos sku=%s candidato_id=%s ventanas=%d restauradas=%d fallidas=%d",
        candidato.sku,
        candidato.candidato_id,
        len(unidades),
        len(restauradas),
        sum(u.fallo is not None for u in unidades),
    )
    return ResultadoPronosticos(candidato, tuple(unidades))


def _pronosticar(
    candidato: Candidato,
    modelo: AdaptadorCandidato,
    reserva: pd.Series,
    ventana: VentanaReserva,
) -> tuple[np.ndarray, FalloPronostico | None]:
    horizonte = len(ventana.fechas)
    invalidas = np.full(horizonte, np.nan)
    try:
        observadas = reserva.loc[: ventana.origen].to_numpy(dtype=float, copy=True)
    except (ValueError, TypeError) as exc:
        return invalidas, _fallo(
            candidato, ventana, "observadas", "observaciones_ausentes", exc
        )
    if not np.isfinite(observadas).all():
        return invalidas, _fallo(
            candidato,
            ventana,
            "observadas",
            "observaciones_ausentes",
            ValueError("faltan observaciones finitas hasta el origen"),
        )
    try:
        valores = np.asarray(modelo.pronosticar(observadas, horizonte), dtype=float)
    except Exception as exc:
        return invalidas, _fallo(
            candidato, ventana, "pronosticar", "pronostico_fallido", exc
        )
    if valores.shape != (horizonte,) or not np.isfinite(valores).all():
        return invalidas, _fallo(
            candidato,
            ventana,
            "valores",
            "pronostico_invalido",
            ValueError(f"se requieren {horizonte} valores finitos en un arreglo 1D"),
        )
    return valores.copy(), None


def _restaurar(
    candidato: Candidato,
    configuracion: ConfiguracionPronosticos,
    ventanas: tuple[VentanaReserva, ...],
    confirmadas: Sequence[UnidadPronostico],
) -> dict[pd.Timestamp, UnidadPronostico]:
    planeadas = {v.origen: v.fechas for v in ventanas}
    restauradas: dict[pd.Timestamp, UnidadPronostico] = {}
    for unidad in confirmadas:
        pronostico = unidad.pronostico
        origen = pronostico.origen
        valores = np.asarray(pronostico.valores, dtype=float)
        fallo = unidad.fallo
        if (
            origen in restauradas
            or origen not in planeadas
            or unidad.candidato != candidato
            or unidad.codigo_version != configuracion.codigo_version
            or unidad.emitido_en.utcoffset() is None
            or not pronostico.fechas.equals(planeadas[origen])
            or valores.shape != (configuracion.horizonte,)
            or (fallo is None and not np.isfinite(valores).all())
            or (
                fallo is not None
                and (
                    fallo.sku != candidato.sku
                    or fallo.candidato_id != candidato.candidato_id
                    or fallo.modelo != candidato.modelo
                    or fallo.origen != origen
                    or np.isfinite(valores).all()
                    or np.isinf(valores).any()
                )
            )
        ):
            _logger.error(
                "Unidad incompatible sku=%s candidato_id=%s origen=%s",
                candidato.sku,
                candidato.candidato_id,
                origen,
            )
            raise ValueError(
                "unidad confirmada incompatible con el protocolo de corrida"
            )
        restauradas[origen] = unidad
    return restauradas


def _fallo(
    candidato: Candidato,
    ventana: VentanaReserva,
    campo: str,
    codigo: CodigoFalloPronostico,
    error: Exception,
) -> FalloPronostico:
    mensaje = (
        f"sku={candidato.sku} candidato_id={candidato.candidato_id} "
        f"modelo={candidato.modelo} origen={ventana.origen.date()} campo={campo}: "
        f"{type(error).__name__}: {error}"
    )
    _logger.error("Fallo de pronostico codigo=%s: %s", codigo, mensaje)
    return FalloPronostico(
        candidato.sku,
        candidato.candidato_id,
        candidato.modelo,
        ventana.origen,
        campo,
        codigo,
        mensaje,
    )

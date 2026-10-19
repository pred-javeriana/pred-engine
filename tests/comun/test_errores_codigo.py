"""Registro unico de codigos de error (G6) y codigos de exclusion de SKU (G7)."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import pandas as pd
import pytest

import pred_engine
from pred_engine.comun import errores_codigo
from pred_engine.comun.errores_codigo import (
    CODIGO_ETAPA_FALLIDA,
    ELEGIBILIDAD_HISTORIA_INSUFICIENTE,
    ELEGIBILIDAD_HUECOS,
    ELEGIBILIDAD_NO_FINITA,
    ELEGIBILIDAD_SIN_HISTORIA,
    ELEGIBILIDAD_SIN_RESERVA,
    codigo_de,
    codigos_conocidos,
)
from pred_engine.comun.reserva import ReserveCut
from pred_engine.optimizacion.router.errores import (
    NingunTrialCompletoError,
    SelectionContractError,
)
from pred_engine.pipeline import (
    Pipeline,
    PipelineExecutionError,
    PipelineResult,
    StageUnavailableError,
)

# Excepciones del motor que a proposito no estan en el catalogo del contrato (no
# llegan a la interfaz o son errores de programacion).
_SIN_CODIGO = {
    "EnsayosIlegiblesError",
    "UnknownProviderError",
    "UnknownModelError",
    "AdaptadorCandidatosError",
    "LoteRechazadoError",
    "ModeloNoRegistradoError",
    "PersistenciaEvidenciaError",
    "EstadoCorridaError",
    "EvidenciaInconsistenteError",
    "EvaluacionRetrospectivaError",
    "ReanudacionError",
    "SelectionError",
    "OutputHandoffError",
    "PipelineExecutionError",
    "LlmTimeoutError",
    "LlmProviderError",
}


def _instancia(clase: type[BaseException]) -> BaseException:
    """Instancia sin llamar al constructor: solo importa el tipo."""
    return clase.__new__(clase)


def _clases_de_errores() -> list[type[BaseException]]:
    clases: dict[str, type[BaseException]] = {}
    for modulo in pkgutil.walk_packages(pred_engine.__path__, "pred_engine."):
        if not modulo.name.endswith(".errores"):
            continue
        for nombre, objeto in inspect.getmembers(
            importlib.import_module(modulo.name), inspect.isclass
        ):
            if (
                issubclass(objeto, BaseException)
                and objeto.__module__ == modulo.name
                and nombre not in _SIN_CODIGO
            ):
                clases[nombre] = objeto
    return sorted(clases.values(), key=lambda clase: clase.__name__)


@pytest.mark.parametrize("clase", _clases_de_errores(), ids=lambda c: c.__name__)
def test_toda_excepcion_tipada_del_motor_tiene_codigo(
    clase: type[BaseException],
) -> None:
    codigo = codigo_de(_instancia(clase))
    assert codigo != CODIGO_ETAPA_FALLIDA, f"{clase.__name__} cae en el respaldo"
    assert codigo in codigos_conocidos()


def test_los_codigos_cumplen_el_patron_del_contrato() -> None:
    import re

    patron = re.compile(r"^[a-z_]+\.[a-z_]+$")
    assert all(patron.match(codigo) for codigo in codigos_conocidos())


def test_las_clases_de_la_tabla_existen() -> None:
    """Una clase renombrada no debe dejar su entrada muerta sin avisar."""
    assert len(errores_codigo._resueltas()) == len(errores_codigo._TABLA)


def test_ningun_trial_es_un_tipo_propio_y_sigue_siendo_de_seleccion() -> None:
    assert issubclass(NingunTrialCompletoError, SelectionContractError)
    assert codigo_de(NingunTrialCompletoError("x")) == "selection.no_viable_trial"
    assert codigo_de(SelectionContractError("ningun trial de HPO")) == (
        "selection.contract"
    )


def test_el_pipeline_se_resuelve_por_la_causa_y_la_etapa() -> None:
    causa = RuntimeError("sin tipo")
    error = PipelineExecutionError("L3", PipelineResult(()), "boom")
    error.__cause__ = causa
    assert codigo_de(error) == "evaluation.failed"

    otra = PipelineExecutionError("L2", PipelineResult(()), "boom")
    otra.__cause__ = NingunTrialCompletoError("x")
    assert codigo_de(otra) == "selection.no_viable_trial"
    assert codigo_de(StageUnavailableError("L4")) == "pipeline.stage_unavailable"


def test_respaldo_por_etapa_para_errores_sin_tipo() -> None:
    assert codigo_de(RuntimeError("x")) == "pipeline.stage_failed"
    assert codigo_de(RuntimeError("x"), etapa="L2") == "pipeline.stage_failed"
    assert codigo_de(RuntimeError("x"), etapa="L3") == "evaluation.failed"


# ------------------------------------------------------------------ elegibilidad
def _pipeline(min_train: int = 5, horizon: int = 3) -> Any:
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.settings = type("S", (), {"min_train": min_train, "horizon": horizon})()
    return pipeline


def _serie(dias: int, *, inicio: str = "2026-01-01") -> pd.DataFrame:
    fechas = pd.date_range(inicio, periods=dias, freq="D")
    return pd.DataFrame({"timestamp": fechas, "demand_qty": [1.0] * dias})


def _reserva(t_star: str) -> ReserveCut:
    corte = pd.Timestamp(t_star)
    return ReserveCut(
        fraction=0.2,
        t_star=corte,
        first_reserved=corte + pd.Timedelta(days=1),
        last_observed=corte + pd.Timedelta(days=2),
        reserved_days=2,
    )


def test_cada_causa_de_exclusion_trae_su_codigo_y_conserva_el_texto() -> None:
    pipeline = _pipeline()
    reserva = _reserva("2026-01-10")
    no_finita = _serie(10)
    no_finita.loc[3, "demand_qty"] = float("nan")

    def codigo(serie: pd.DataFrame, motor: Any = pipeline) -> str | None:
        resultado = motor._exclusion_cause(serie, reserva)
        return None if resultado is None else resultado[0]

    assert codigo(_serie(0)) == ELEGIBILIDAD_SIN_HISTORIA
    assert codigo(_serie(10).drop(index=4)) == ELEGIBILIDAD_HUECOS
    assert codigo(_serie(9)) == ELEGIBILIDAD_SIN_RESERVA
    assert codigo(no_finita) == ELEGIBILIDAD_NO_FINITA
    assert codigo(_serie(10), _pipeline(min_train=20)) == (
        ELEGIBILIDAD_HISTORIA_INSUFICIENTE
    )
    assert codigo(_serie(10)) is None

    _, causa = _pipeline(min_train=20)._exclusion_cause(_serie(10), reserva)
    assert causa.startswith("historia admisible insuficiente: 10 dias")

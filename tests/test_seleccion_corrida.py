"""``seleccion.jsonl`` y la lectura de ensayos sin Optuna (G2)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pred_engine.comun.dataclasses.validacion_temporal import (
    ResultadoVentana,
    ResultadoWalkForward,
    VentanaTemporal,
)
from pred_engine.comun.ejecucion_paralela import RegistroUnidad
from pred_engine.comun.ensayos_hpo import (
    EnsayosIlegiblesError,
    buscar_estudio,
    ensayo_a_dict,
    leer_ensayos,
)
from pred_engine.comun.reserva import ReserveCut
from pred_engine.optimizacion.optimizadores.modelos_fundacionales.estrategia import (
    FoundationSelectionStrategy,
)
from pred_engine.optimizacion.router import SelectionRequest, SelectionResult
from pred_engine.optimizacion.router.politica import PROFILE_BY_SKU_CLASS
from pred_engine.pipeline import (
    PRINCIPAL_STAGES,
    EvaluatedCandidate,
    EvaluationArtifact,
    EvaluationSettings,
    FittedCandidate,
    FittingArtifact,
    IngestionArtifact,
    PipelineInput,
    PipelineResult,
    SkuExclusion,
    StageExecution,
    UnitExecution,
)
from pred_engine.pipeline_setup import ConfiguredTopologyPolicy, build_pipeline
from pred_engine.run_artifacts import write_run
from pred_engine.run_seleccion import (
    ARCHIVO_SELECCION,
    SELECCION_SCHEMA,
    construir_seleccion,
    escribir_seleccion,
)

SESION = "r-prueba"


# ----------------------------------------------------------------- lectura de ensayos
def _linea(
    estado: str,
    numero: int,
    *,
    valor: float | None = 1.5,
    motivo: str | None = None,
    familia: str = "classical",
    sku: str = "S1",
) -> str:
    atributos: dict[str, Any] = {
        "trial_id": f"{familia}-{sku}-{numero:04d}",
        "timestamp": f"2026-10-14T16:19:0{numero}+00:00",
        "familia": familia,
        "sku_id": sku,
        "n_ventanas": numero + 1,
    }
    if motivo is not None:
        atributos["motivo"] = motivo
    return json.dumps(
        {
            "estado_optuna": estado,
            "params": {"p": numero, "d": 1},
            "valor": valor,
            "user_attrs": atributos,
            "intermediate_values": {},
        }
    )


def _estudio(raiz: Path, nombre: str, lineas: list[str], **manifiesto: Any) -> Path:
    carpeta = raiz / nombre
    carpeta.mkdir(parents=True)
    (carpeta / "backend.jsonl").write_text("".join(f"{x}\n" for x in lineas))
    (carpeta / "manifiesto.json").write_text(
        json.dumps(
            {"familia": "classical", "sku_id": "S1", "metrica_objetivo": "mae"}
            | manifiesto
        )
    )
    return carpeta


def test_cada_estado_del_backend_se_lee_con_el_vocabulario_del_motor(tmp_path):
    carpeta = _estudio(
        tmp_path,
        "classical-S1-x",
        [
            _linea("COMPLETE", 0),
            _linea("PRUNED", 1, motivo="poda_semantica:prediccion_nula"),
            _linea("FAIL", 2, valor=None, motivo="ventana=1 ajuste_fallido: X: y"),
            _linea("RUNNING", 3, valor=None),
            _linea("WAITING", 4, valor=None),
        ],
    )
    ensayos = leer_ensayos(carpeta)
    assert [t.estado for t in ensayos] == [
        "completado",
        "podado",
        "fallido",
        "corriendo",
        "pendiente",
    ]
    podado, fallido = ensayos[1], ensayos[2]
    assert podado.motivo == "poda_semantica:prediccion_nula"
    assert (
        fallido.motivo.startswith("ventana=1 ajuste_fallido") and fallido.valor is None
    )
    assert ensayos[0].motivo is None
    assert (ensayos[0].id, ensayos[0].n_ventanas) == ("classical-S1-0000", 1)
    assert ensayos[0].configuracion == {"p": 0, "d": 1}
    assert ensayos[0].metrica_objetivo == "mae" and ensayos[0].familia == "classical"


def test_el_ensayo_tiene_la_forma_de_trialrow_del_contrato(tmp_path):
    carpeta = _estudio(tmp_path, "e", [_linea("PRUNED", 0, motivo="m")])
    assert ensayo_a_dict(leer_ensayos(carpeta)[0]) == {
        "id": "classical-S1-0000",
        "configuracion": {"p": 0, "d": 1},
        "estado": "podado",
        "valor": 1.5,
        "n_ventanas": 1,
        "motivo": "m",
        "timestamp": "2026-10-14T16:19:00+00:00",
    }


def test_un_valor_no_finito_es_nulo_y_sin_trial_id_se_numera_por_carpeta(tmp_path):
    linea = json.loads(_linea("COMPLETE", 0, valor=None))
    linea["valor"] = float("inf")
    del linea["user_attrs"]["trial_id"]
    carpeta = _estudio(tmp_path, "ml-S1-x", [json.dumps(linea)])
    (ensayo,) = leer_ensayos(carpeta)
    assert ensayo.valor is None and ensayo.id == "ml-S1-x-0000"


def test_sin_archivo_no_hay_ensayos(tmp_path):
    assert leer_ensayos(tmp_path / "no-existe") == ()
    (tmp_path / "vacio").mkdir()
    assert leer_ensayos(tmp_path / "vacio") == ()


@pytest.mark.parametrize(
    "linea", ["{no es json", json.dumps({"params": {}}), _linea("ENIGMA", 0)]
)
def test_una_linea_ilegible_no_se_omite_en_silencio(tmp_path, linea):
    carpeta = _estudio(tmp_path, "e", [_linea("COMPLETE", 0), linea])
    with pytest.raises(EnsayosIlegiblesError, match="backend.jsonl:2"):
        leer_ensayos(carpeta)


def test_el_estudio_se_busca_por_manifiesto_porque_dl_no_lleva_el_sku_en_el_nombre(
    tmp_path,
):
    _estudio(tmp_path, "classical-S1-x", [], familia="classical", sku_id="S1")
    dl = _estudio(tmp_path, "dl-" + "a" * 64, [], familia="dl", sku_id="S2")
    assert buscar_estudio(tmp_path, "dl", "S2") == dl
    assert buscar_estudio(tmp_path, "dl", "S1") is None
    assert buscar_estudio(tmp_path / "no-existe", "dl", "S2") is None


def test_leer_ensayos_no_importa_optuna():
    codigo = (
        "import sys, pred_engine.comun.ensayos_hpo as m;"
        "assert 'optuna' not in sys.modules, 'se importo optuna'"
    )
    subprocess.run([sys.executable, "-I", "-c", codigo], check=True)


# ------------------------------------------ un resultado armado a mano (sin modelos)
# Estas pruebas no ajustan nada: el resultado de L2 y L3 se construye a mano, asi que
# el estado de cada familia y los numeros esperados no dependen de la plataforma
# (BLAS, LightGBM...).
T_STAR = pd.Timestamp("2026-03-13")
LARGO = 100  # dias de historia de cada SKU hasta t*
CLASES = {"liso": "smooth", "irregular": "lumpy", "corto": "smooth", "tardio": "smooth"}


def _seleccion(sku: str, familia: str, **payload: Any) -> SelectionResult:
    base: dict[str, Any] = {
        "metrica_objetivo": "mae",
        "valor": 1.5,
        "n_ventanas": 2,
        "n_trials": 3,
        "n_completados": 1,
        "n_podados": 1,
        "n_fallidos": 1,
        "seed": 7,
        "estudio_hpo": f"{familia}-{sku}-s",
    }
    extra: dict[str, Any] = {
        "classical": {"order": [1, 1, 1], "seasonal_order": [0, 1, 1, 7]},
        "ml": {"hiperparametros": {"lags": 7}},
        "foundation": {"configuracion": {"modelo": "chronos-2"}, "optimizado": False},
    }[familia]
    contenido = extra if familia == "foundation" else {**base, **extra}
    return SelectionResult(
        sku_id=sku,
        sku_class=CLASES[sku],  # type: ignore[arg-type]
        family=familia,  # type: ignore[arg-type]
        profile=PROFILE_BY_SKU_CLASS[CLASES[sku]],  # type: ignore[index]
        produced_by=f"{familia.capitalize()}SelectionStrategy",
        policy_version="2.2.0-initial:configured:classical,ml,foundation",
        payload={**contenido, **payload},
        forecast_config={"m": 7},
        forecast_seed=7,
    )


def _candidato(sku: str, familia: str) -> FittedCandidate:
    return FittedCandidate(
        selection=_seleccion(sku, familia),
        series=np.zeros(LARGO),
        forecast=np.zeros(18),
    )


def _walk_forward(valor: float = 1.5) -> ResultadoWalkForward:
    ventanas = tuple(
        ResultadoVentana(
            ventana=VentanaTemporal(i, 0, 60 + 7 * i, 60 + 7 * i, 67 + 7 * i),
            metricas={"mae": valor + i, "rmse": valor + 1 + i, "smape": float("nan")},
            y_train=np.zeros(3),
            y_real=np.zeros(7),
            y_pred=np.zeros(7),
            duracion_s=0.25,
            fallo=None,
        )
        for i in range(2)
    )
    return ResultadoWalkForward(ventanas, "mae", valor, True, 2, 2)


def _unidad(
    etapa: str, sku: str, familia: str, error: str | None = None
) -> UnitExecution:
    registro = RegistroUnidad(
        estado="fallida" if error else "completada",
        pid=1,
        inicio=0.0,
        fin=1.0,
        error=error,
    )
    return UnitExecution(etapa, sku, familia, registro)  # type: ignore[arg-type]


def _etapas(
    fitting: Any, evaluation: Any, *, mensaje_l3: str = ""
) -> tuple[StageExecution, ...]:
    ingestion = fitting.ingestion if fitting else None
    salidas = (ingestion, fitting, evaluation, None)
    estados = (
        "completed",
        "completed" if fitting else "failed",
        "completed",
        "blocked",
    )
    return tuple(
        StageExecution(d, state=e, output=o, message=mensaje_l3 if d.id == "L3" else "")  # type: ignore[arg-type]
        for d, e, o in zip(PRINCIPAL_STAGES, estados, salidas, strict=True)
    )


def _resultado(
    *,
    familias: tuple[str, ...] = ("classical", "ml", "foundation"),
    sin_l3: bool = False,
) -> PipelineResult:
    """Cuatro SKU, cada uno con un destino distinto en L2 y L3.

    * ``liso`` (smooth): clasica y ML evaluadas con exito; la fundacional, evaluada.
    * ``irregular`` (lumpy): la clasica falla en L2 (sin candidato); la matriz no le
      asigna ML.
    * ``tardio`` (smooth): la clasica se ajusta en L2 pero falla en L3.
    * ``corto`` (smooth): el motor lo excluyo antes de L2.
    """
    panel = pd.DataFrame(
        {
            "sku_id": list(CLASES),
            "sku_class": list(CLASES.values()),
            "timestamp": [T_STAR] * len(CLASES),
        }
    )
    candidatos = (
        _candidato("liso", "classical"),
        _candidato("liso", "ml"),
        _candidato("liso", "foundation"),
        _candidato("tardio", "classical"),
    )
    candidatos = tuple(c for c in candidatos if c.selection.family in familias)
    unidades_l2 = tuple(
        _unidad("L2", c.selection.sku_id, c.selection.family) for c in candidatos
    ) + (
        _unidad(
            "L2",
            "irregular",
            "classical",
            "SelectionContractError: ningun trial de HPO completo para "
            "sku_id='irregular' (0 fallidos, 2 podados)",
        ),
    )
    fitting = FittingArtifact(
        ingestion=IngestionArtifact(Path("panel.parquet"), panel),
        reserve=ReserveCut(
            T_STAR, T_STAR + pd.Timedelta(days=1), T_STAR + pd.Timedelta(days=18), 18
        ),
        candidates=candidatos,
        units=unidades_l2,
        exclusions=(
            SkuExclusion(
                "corto",
                "la serie termina antes de t*: no tiene observaciones reservadas",
            ),
        ),
        policy=ConfiguredTopologyPolicy(("classical", "ml", "foundation")),
    )
    if sin_l3:
        return PipelineResult(
            _etapas(fitting, None, mensaje_l3="0 de 4 unidades evaluaron")
        )
    evaluados = tuple(
        EvaluatedCandidate(c, _walk_forward(1.5 + i))
        for i, c in enumerate(candidatos)
        if c.selection.sku_id == "liso"
    )
    evaluacion = EvaluationArtifact(
        fitting=fitting,
        settings=EvaluationSettings(40, horizon=7, step=7, metric="mae"),
        candidates=evaluados,
        units=tuple(
            _unidad("L3", c.fitted.selection.sku_id, c.fitted.selection.family)
            for c in evaluados
        )
        + (
            _unidad(
                "L3",
                "tardio",
                "classical",
                "EvaluacionError: ninguna ventana evaluable",
            ),
        ),
    )
    return PipelineResult(_etapas(fitting, evaluacion))


def _hpo(raiz: Path) -> Path:
    """La carpeta ``hpo/`` de la corrida: un estudio por unidad que llego a tenerlo."""
    hpo = raiz / "hpo"
    lineas = [
        _linea("COMPLETE", 0, sku="liso"),
        _linea("PRUNED", 1, motivo="escalon=4 valor=2.5", sku="liso"),
        _linea(
            "FAIL", 2, valor=None, motivo="ventana=1 ajuste_fallido: X: y", sku="liso"
        ),
    ]
    _estudio(hpo, "classical-liso-s", lineas, sku_id="liso")
    _estudio(
        hpo,
        "ml-liso-s",
        [_linea("COMPLETE", 0, familia="ml", sku="liso")],
        familia="ml",
        sku_id="liso",
    )
    _estudio(hpo, "classical-tardio-s", lineas, sku_id="tardio")
    # Sin `estudio_hpo` en el payload (la unidad fallo): solo se halla por manifiesto.
    _estudio(
        hpo,
        "classical-irregular-s",
        [
            _linea(
                "PRUNED", i, motivo="poda_semantica:prediccion_nula", sku="irregular"
            )
            for i in range(2)
        ],
        sku_id="irregular",
    )
    return hpo


@pytest.fixture
def documentos(tmp_path):
    lineas = construir_seleccion(_resultado(), run_id="r-1", hpo=_hpo(tmp_path))
    return {linea["sku_id"]: linea for linea in lineas}


def _familias(documento: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f["family"]: f for f in documento["families"]}


def test_una_linea_por_sku_con_las_cuatro_familias_en_orden(documentos):
    assert list(documentos) == ["corto", "irregular", "liso", "tardio"]
    for linea in documentos.values():
        assert linea["schema"] == SELECCION_SCHEMA and linea["run_id"] == "r-1"
        assert [f["family"] for f in linea["families"]] == [
            "classical",
            "ml",
            "dl",
            "foundation",
        ]
    liso = documentos["liso"]
    assert (liso["sku_class"], liso["profile"]) == ("smooth", "dense_stable")
    assert liso["policy_version"] == "2.2.0-initial:configured:classical,ml,foundation"
    assert documentos["irregular"]["profile"] == "sparse_variable"


def test_una_familia_exitosa_trae_evidencia_ensayos_y_ventanas(documentos):
    clasica = _familias(documentos["liso"])["classical"]

    assert (clasica["estado"], clasica["modelo"]) == ("exitosa", "classical:sarima")
    assert clasica["produced_by"] == "ClassicalSelectionStrategy"
    assert clasica["evidence"]["family"] == "classical"
    assert clasica["evidence"]["order"] == [1, 1, 1]
    assert clasica["forecast_config"] == {"m": 7} and clasica["forecast_seed"] == 7
    assert [t["estado"] for t in clasica["trials"]] == [
        "completado",
        "podado",
        "fallido",
    ]
    assert clasica["trials"][1]["motivo"] == "escalon=4 valor=2.5"
    assert clasica["trials"][2]["motivo"].startswith("ventana=1 ajuste_fallido")
    assert clasica["trials"][0]["id"] == "classical-liso-0000"
    assert "error" not in clasica and "exclusion" not in clasica


def test_el_walk_forward_resume_y_cada_ventana_con_su_corte(documentos):
    resumen = _familias(documentos["liso"])["classical"]["walk_forward"]

    assert resumen["metrica_objetivo"] == "mae" and resumen["valor_agregado"] == 1.5
    assert (resumen["completo"], resumen["n_ventanas_evaluadas"]) == (True, 2)
    assert resumen["selection_scope"] == "same_history"
    assert resumen["statistical_comparison_available"] is False
    primera, segunda = resumen["windows"]
    # El primer dia de validacion: la historia empieza LARGO - 1 dias antes de t*.
    inicio = T_STAR - pd.Timedelta(days=LARGO - 1)
    assert primera["corte"] == (inicio + pd.Timedelta(days=60)).date().isoformat()
    assert segunda["corte"] == (inicio + pd.Timedelta(days=67)).date().isoformat()
    assert primera["corte"] == "2026-02-02"
    assert (primera["indice"], primera["inicio_train"], primera["fin_train"]) == (
        0,
        0,
        60,
    )
    assert (primera["inicio_val"], primera["fin_val"]) == (60, 67)
    assert primera["metricas"] == {
        "mae": 1.5,
        "rmse": 2.5,
        "smape": None,
    }  # NaN -> null
    assert primera["duracion_s"] == 0.25 and primera["fallo"] is None


def test_una_unidad_fallida_en_l2_conserva_su_error_y_los_ensayos_que_la_explican(
    documentos,
):
    clasica = _familias(documentos["irregular"])["classical"]

    assert (clasica["estado"], clasica["modelo"]) == ("fallida", "classical:sarima")
    assert clasica["error"]["stage"] == "L2"
    assert clasica["error"]["text"].startswith("SelectionContractError: ningun trial")
    assert "evidence" not in clasica and "walk_forward" not in clasica
    # Sin `estudio_hpo` en el payload: el estudio se encuentra por su manifiesto.
    assert [t["estado"] for t in clasica["trials"]] == ["podado", "podado"]
    assert all("poda_semantica" in t["motivo"] for t in clasica["trials"])


def test_una_unidad_fallida_en_l3_conserva_la_evidencia_de_l2(documentos):
    clasica = _familias(documentos["tardio"])["classical"]

    assert clasica["estado"] == "fallida"
    assert clasica["error"] == {
        "stage": "L3",
        "text": "EvaluacionError: ninguna ventana evaluable",
    }
    assert clasica["evidence"]["family"] == "classical" and clasica["produced_by"]
    assert len(clasica["trials"]) == 3 and "walk_forward" not in clasica


def test_si_l3_falla_entera_la_causa_es_el_mensaje_de_la_etapa():
    lineas = construir_seleccion(_resultado(sin_l3=True), run_id="r-1", hpo="no-existe")
    liso = _familias({x["sku_id"]: x for x in lineas}["liso"])

    assert liso["ml"]["estado"] == "fallida"
    assert liso["ml"]["error"] == {"stage": "L3", "text": "0 de 4 unidades evaluaron"}
    assert liso["ml"]["trials"] is None  # sin carpeta hpo/ no hay estudio persistido


def test_la_matriz_y_la_configuracion_excluyen_familias_con_su_motivo(documentos):
    liso = _familias(documentos["liso"])
    assert liso["dl"] == {
        "family": "dl",
        "estado": "excluida",
        "exclusion": {
            "reason_code": "not_configured",
            "detail": "La corrida se configuro solo con las familias "
            "classical, ml, foundation.",
        },
    }
    irregular = _familias(documentos["irregular"])
    assert irregular["ml"]["exclusion"] == {
        "reason_code": "not_in_policy_matrix",
        "detail": "La politica 2.2.0-initial no asigna ml a la clase lumpy.",
    }
    assert irregular["dl"]["exclusion"]["reason_code"] == "not_in_policy_matrix"


def test_con_una_sola_familia_configurada_el_motivo_va_en_singular(tmp_path):
    resultado = _resultado(familias=("classical",))
    fitting = resultado.fitting
    assert fitting is not None
    solo = FittingArtifact(
        fitting.ingestion,
        fitting.reserve,
        fitting.candidates,
        fitting.units,
        fitting.exclusions,
        ConfiguredTopologyPolicy(("classical",)),
    )
    resultado = PipelineResult(_etapas(solo, resultado.evaluation))
    lineas = construir_seleccion(resultado, run_id="r", hpo=tmp_path)
    ml = _familias({x["sku_id"]: x for x in lineas}["liso"])["ml"]

    assert ml["exclusion"]["reason_code"] == "not_configured"
    assert (
        ml["exclusion"]["detail"]
        == "La corrida se configuro solo con la familia classical."
    )


def test_un_sku_excluido_por_el_motor_no_es_ejecutable_con_su_causa(documentos):
    corto = _familias(documentos["corto"])
    for familia in ("classical", "ml", "foundation"):
        assert corto[familia] == {
            "family": familia,
            "estado": "no_ejecutable",
            "cause": "la serie termina antes de t*: no tiene observaciones reservadas",
        }
    assert (
        corto["dl"]["estado"] == "excluida"
    )  # la matriz le deja DL, pero no se configuro


def test_la_familia_fundacional_no_tiene_ensayos_ni_contadores(documentos):
    fundacional = _familias(documentos["liso"])["foundation"]

    assert fundacional["estado"] == "exitosa"
    assert fundacional["trials"] is None
    assert fundacional["evidence"] == {
        "family": "foundation",
        "configuracion": {"modelo": "chronos-2"},
        "optimizado": False,
    }
    assert fundacional["walk_forward"]["windows"]


def test_la_familia_fundacional_declara_que_no_optimiza():
    resultado = FoundationSelectionStrategy().select(
        SelectionRequest(sku_id="s", sku_class="smooth"), "dense_stable"
    )
    # Sin estudio no hay ensayos ni contadores: el exportador deja `trials` nulo.
    assert resultado.payload["optimizado"] is False
    assert not {"n_trials", "estudio_hpo"} & set(resultado.payload)


def test_el_archivo_es_json_estricto_con_una_linea_por_sku_y_se_reescribe_igual(
    tmp_path,
):
    resultado = _resultado()
    hpo = _hpo(tmp_path)
    ruta = escribir_seleccion(resultado, tmp_path, run_id="r-1", hpo=hpo)

    assert ruta == tmp_path / ARCHIVO_SELECCION
    texto = ruta.read_text()
    lineas = texto.splitlines()
    assert len(lineas) == 4
    for linea in lineas:
        json.loads(linea, parse_constant=_prohibida)
    assert escribir_seleccion(resultado, tmp_path, run_id="r-1", hpo=hpo) == ruta
    assert ruta.read_text() == texto


def test_sin_ajuste_en_l2_no_hay_seleccion_que_escribir(tmp_path):
    sin_l2 = PipelineResult(_etapas(None, None))
    assert construir_seleccion(sin_l2, run_id="r", hpo=tmp_path) == []
    assert escribir_seleccion(sin_l2, tmp_path, run_id="r", hpo=tmp_path) is None
    assert not (tmp_path / ARCHIVO_SELECCION).exists()


def _prohibida(constante: str) -> float:
    raise AssertionError(f"JSON no estricto: {constante}")


# ----------------------------------------------- corrida real (ajusta modelos: lenta)
def _serie(n: int, valores: list[float]) -> list[float]:
    return [valores[i % len(valores)] for i in range(n)]


def _panel_csv(raiz: Path) -> Path:
    n = 120
    fechas = pd.date_range("2024-01-01", periods=n)
    suave = [20 + 0.35 * i + (i % 5 - 2) for i in range(n)]
    irregular = _serie(n, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 30.0])
    frames = [
        pd.DataFrame(
            {"sku_id": sku, "timestamp": fechas[:largo], "demand_qty": valores[:largo]}
        )
        for sku, valores, largo in (
            ("liso", suave, n),
            ("irregular", irregular, n),
            ("corto", suave, 10),
        )
    ]
    panel = pd.concat(frames)
    panel["lead_time_days"] = 3
    ruta = raiz / "ventas.csv"
    panel.to_csv(ruta, index=False)
    return ruta


@pytest.mark.slow
def test_una_corrida_real_deja_una_seleccion_coherente_con_sus_archivos(tmp_path):
    """Integracion con modelos reales.

    El resultado numerico depende de la plataforma (en Linux una familia que en macOS
    fallaba converge), asi que solo se afirma lo que vale para cualquier resultado.
    """
    destino = tmp_path / "runs" / SESION
    pipeline = build_pipeline(
        EvaluationSettings(40, horizon=2, step=14, seasonality=7, metric="mae"),
        families=("classical", "ml"),
        n_trials=2,
        seed=17,
        hpo_root=destino / "hpo",
        session=SESION,
    )
    resultado = pipeline.run(
        PipelineInput(csv_path=_panel_csv(tmp_path), data_root=tmp_path / "data")
    )
    archivos = write_run(
        resultado, destino, run_id=SESION, metadata={"inputs": {}, "settings": {}}
    )
    texto = archivos["seleccion"].read_text()
    documentos = {x["sku_id"]: x for x in map(json.loads, texto.splitlines())}
    evaluacion = pd.read_parquet(destino / "evaluacion.parquet")

    assert list(documentos) == ["corto", "irregular", "liso"]
    assert json.loads((destino / "corrida.json").read_text())["files"]["seleccion"] == (
        "seleccion.jsonl"
    )
    assert all(
        e["estado"] == "no_ejecutable"
        for e in _familias(documentos["corto"]).values()
        if e["family"] in ("classical", "ml")
    )
    assert _familias(documentos["irregular"])["ml"]["estado"] == "excluida"
    for sku, documento in documentos.items():
        for familia, entrada in _familias(documento).items():
            estado = entrada["estado"]
            assert estado in {"exitosa", "fallida", "no_ejecutable", "excluida"}
            if estado == "exitosa":
                fila = evaluacion[
                    (evaluacion.sku_id == sku) & (evaluacion.familia == familia)
                ].iloc[0]
                resumen = entrada["walk_forward"]
                assert resumen["valor_agregado"] == pytest.approx(
                    fila["valor_agregado"]
                )
                assert resumen["n_ventanas_evaluadas"] == fila["n_ventanas"]
                assert len(resumen["windows"]) == resumen["n_ventanas_evaluadas"]
                evidencia = entrada["evidence"]
                assert evidencia["n_trials"] == len(entrada["trials"])
                assert (
                    evidencia["n_completados"]
                    + evidencia["n_podados"]
                    + (evidencia["n_fallidos"])
                    == evidencia["n_trials"]
                )
            elif estado == "fallida":
                assert (
                    entrada["error"]["stage"] in {"L2", "L3"}
                    and entrada["error"]["text"]
                )
                assert entrada["trials"] is None or all(
                    t["motivo"]
                    for t in entrada["trials"]
                    if t["estado"] != "completado"
                )
    write_run(
        resultado, destino, run_id=SESION, metadata={"inputs": {}, "settings": {}}
    )
    assert archivos["seleccion"].read_text() == texto

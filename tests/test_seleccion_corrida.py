"""``seleccion.jsonl`` y la lectura de ensayos sin Optuna (G2)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pred_engine.comun.ensayos_hpo import (
    EnsayosIlegiblesError,
    buscar_estudio,
    ensayo_a_dict,
    leer_ensayos,
)
from pred_engine.optimizacion.optimizadores.modelos_fundacionales.estrategia import (
    FoundationSelectionStrategy,
)
from pred_engine.optimizacion.router import SelectionRequest
from pred_engine.pipeline import EvaluationSettings, PipelineInput
from pred_engine.pipeline_setup import build_pipeline
from pred_engine.run_artifacts import write_run
from pred_engine.run_seleccion import SELECCION_SCHEMA

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


# ------------------------------------------------------------------ corrida completa
def _serie(n: int, valores: list[float]) -> list[float]:
    return [valores[i % len(valores)] for i in range(n)]


def _panel_csv(raiz: Path) -> Path:
    n = 120
    fechas = pd.date_range("2024-01-01", periods=n)
    suave = [20 + 0.35 * i + (i % 5 - 2) for i in range(n)]
    # Intermitente y muy variable: la matriz de politica no le asigna ml ni dl.
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


@pytest.fixture(scope="module")
def corrida(tmp_path_factory) -> dict[str, Any]:
    raiz = tmp_path_factory.mktemp("corrida")
    destino = raiz / "runs" / SESION
    pipeline = build_pipeline(
        EvaluationSettings(40, horizon=2, step=14, seasonality=7, metric="mae"),
        families=("classical", "ml"),
        n_trials=2,
        seed=17,
        hpo_root=destino / "hpo",
        session=SESION,
    )
    resultado = pipeline.run(
        PipelineInput(csv_path=_panel_csv(raiz), data_root=raiz / "data")
    )
    archivos = write_run(
        resultado, destino, run_id=SESION, metadata={"inputs": {}, "settings": {}}
    )
    lineas = [json.loads(x) for x in archivos["seleccion"].read_text().splitlines()]
    return {
        "resultado": resultado,
        "destino": destino,
        "archivos": archivos,
        "skus": {x["sku_id"]: x for x in lineas},
        "lineas": lineas,
    }


def _familias(documento: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f["family"]: f for f in documento["families"]}


def test_una_linea_por_sku_con_las_cuatro_familias_en_orden(corrida):
    assert list(corrida["skus"]) == ["corto", "irregular", "liso"]
    for linea in corrida["lineas"]:
        assert linea["schema"] == SELECCION_SCHEMA and linea["run_id"] == SESION
        assert [f["family"] for f in linea["families"]] == [
            "classical",
            "ml",
            "dl",
            "foundation",
        ]
    assert corrida["archivos"]["seleccion"].name == "seleccion.jsonl"
    registrado = json.loads((corrida["destino"] / "corrida.json").read_text())
    assert registrado["files"]["seleccion"] == "seleccion.jsonl"


def test_el_sku_liso_trae_evidencia_ensayos_y_ventanas_de_cada_familia(corrida):
    liso = corrida["skus"]["liso"]
    assert (liso["sku_class"], liso["profile"]) == ("smooth", "dense_stable")
    assert liso["policy_version"].startswith("2.2.0-initial:configured:")
    familias = _familias(liso)
    for nombre, modelo in (("classical", "classical:sarima"), ("ml", "ml:lightgbm")):
        entrada = familias[nombre]
        assert (entrada["estado"], entrada["modelo"]) == ("exitosa", modelo)
        evidencia = entrada["evidence"]
        assert evidencia["family"] == nombre
        assert evidencia["n_trials"] == len(entrada["trials"]) == 2
        assert evidencia["estudio_hpo"] == f"{nombre}-liso-{SESION}"
        contados = {
            estado: sum(t["estado"] == estado for t in entrada["trials"])
            for estado in ("completado", "podado", "fallido")
        }
        assert contados == {
            "completado": evidencia["n_completados"],
            "podado": evidencia["n_podados"],
            "fallido": evidencia["n_fallidos"],
        }
        assert entrada["forecast_seed"] == 17 and entrada["forecast_config"]
        resumen = entrada["walk_forward"]
        assert resumen["metrica_objetivo"] == "mae" and resumen["completo"]
        assert resumen["selection_scope"] == "same_history"
        assert resumen["statistical_comparison_available"] is False
        ventanas = resumen["windows"]
        assert len(ventanas) == resumen["n_ventanas_evaluadas"] > 0
        assert [v["indice"] for v in ventanas] == list(range(len(ventanas)))
        primera = ventanas[0]
        assert primera["fin_train"] == primera["inicio_val"]
        assert set(primera["metricas"]) == {"mae", "rmse", "smape", "mase"}
        assert primera["fallo"] is None and primera["corte"] > "2024"


def test_las_ventanas_coinciden_con_los_parquet_de_la_corrida(corrida):
    destino = corrida["destino"]
    cruda = pd.read_parquet(destino / "walk_forward.parquet")
    evaluacion = pd.read_parquet(destino / "evaluacion.parquet")
    for sku, documento in corrida["skus"].items():
        for familia, entrada in _familias(documento).items():
            if entrada["estado"] != "exitosa":
                continue
            fila = evaluacion[
                (evaluacion.sku_id == sku) & (evaluacion.familia == familia)
            ].iloc[0]
            resumen = entrada["walk_forward"]
            assert resumen["valor_agregado"] == pytest.approx(fila["valor_agregado"])
            assert resumen["n_ventanas_evaluadas"] == fila["n_ventanas"]
            dias = cruda[(cruda.sku_id == sku) & (cruda.familia == familia)]
            primera = dias[dias.ventana == 0].timestamp.min()
            assert resumen["windows"][0]["corte"] == primera.date().isoformat()


def test_la_matriz_y_la_configuracion_excluyen_familias_con_su_motivo(corrida):
    liso = _familias(corrida["skus"]["liso"])
    assert liso["dl"] == {
        "family": "dl",
        "estado": "excluida",
        "exclusion": {
            "reason_code": "not_configured",
            "detail": "La corrida se configuro solo con las familias classical, ml.",
        },
    }
    assert liso["foundation"]["exclusion"]["reason_code"] == "not_configured"
    irregular = _familias(corrida["skus"]["irregular"])
    assert corrida["skus"]["irregular"]["sku_class"] in {"intermittent", "lumpy"}
    assert irregular["dl"]["exclusion"] == {
        "reason_code": "not_in_policy_matrix",
        "detail": "La politica 2.2.0-initial no asigna dl a la clase "
        f"{corrida['skus']['irregular']['sku_class']}.",
    }


def test_un_sku_excluido_por_el_motor_no_es_ejecutable_con_su_causa(corrida):
    corto = _familias(corrida["skus"]["corto"])
    for familia in ("classical", "ml"):
        assert corto[familia] == {
            "family": familia,
            "estado": "no_ejecutable",
            "cause": "la serie termina antes de t*: no tiene observaciones reservadas",
        }
    assert corto["dl"]["estado"] == "excluida"


def test_una_unidad_fallida_conserva_su_error_y_los_ensayos_que_la_explican(corrida):
    clasica = _familias(corrida["skus"]["irregular"])["classical"]
    assert clasica["estado"] == "fallida" and clasica["modelo"] == "classical:sarima"
    assert clasica["error"]["stage"] == "L2"
    assert clasica["error"]["text"].startswith("SelectionContractError: ningun trial")
    assert "evidence" not in clasica and "walk_forward" not in clasica
    # Podados por una regla, no fallidos: el motivo dice cual.
    assert [t["estado"] for t in clasica["trials"]] == ["podado", "podado"]
    assert all("poda_semantica" in t["motivo"] for t in clasica["trials"])


def test_la_familia_fundacional_declara_que_no_optimiza():
    resultado = FoundationSelectionStrategy().select(
        SelectionRequest(sku_id="s", sku_class="smooth"), "dense_stable"
    )
    # Sin estudio no hay ensayos ni contadores: el exportador deja `trials` nulo.
    assert resultado.payload["optimizado"] is False
    assert not {"n_trials", "estudio_hpo"} & set(resultado.payload)


def test_el_archivo_es_json_estricto_y_se_reescribe_igual(corrida):
    texto = corrida["archivos"]["seleccion"].read_text()
    for linea in texto.splitlines():
        json.loads(linea, parse_constant=_prohibida)
    write_run(
        corrida["resultado"],
        corrida["destino"],
        run_id=SESION,
        metadata={"inputs": {}, "settings": {}},
    )
    assert corrida["archivos"]["seleccion"].read_text() == texto


def _prohibida(constante: str) -> float:
    raise AssertionError(f"JSON no estricto: {constante}")

"""Componente 9 (parcial): 'podado' y 'fallido' deben ser distinguibles, con motivo."""

from __future__ import annotations

import json
from pathlib import Path

from pred_engine.optimizacion.optimizadores.HPO.registro import RegistroEstudio


def _ventanas():
    from pred_engine.comun.dataclasses.validacion_temporal import VentanaTemporal

    return (
        VentanaTemporal(
            indice=0, inicio_train=0, fin_train=10, inicio_val=10, fin_val=11
        ),
    )


def test_podado_y_fallido_son_estados_distintos_con_motivo_no_nulo():
    registro = RegistroEstudio(
        familia="prueba", metrica_objetivo="mase", ventanas=_ventanas()
    )

    registro.abrir("t1", {"p": 1})
    registro.actualizar("t1", n_evaluadas=4, valor_parcial=2.0)
    registro.podar("t1", ventana=4, valor=2.0, motivo="asha")

    registro.abrir("t2", {"p": 2})
    registro.fallar("t2", motivo="excepcion_de_ajuste")

    trials = {t.id: t for t in registro.trials}
    assert trials["t1"].estado == "podado"
    assert trials["t1"].motivo is not None and "asha" in trials["t1"].motivo
    assert trials["t2"].estado == "fallido"
    assert trials["t2"].motivo == "excepcion_de_ajuste"
    assert trials["t1"].estado != trials["t2"].estado


def test_jsonl_round_trip(tmp_path: Path):
    registro = RegistroEstudio(
        familia="prueba", metrica_objetivo="mase", ventanas=_ventanas()
    )
    registro.abrir("t1", {"p": 1})
    registro.completar("t1", valor=0.5, n_ventanas=10)
    ruta = tmp_path / "estudio.jsonl"
    registro.volcar_jsonl(ruta)

    lineas = ruta.read_text(encoding="utf-8").strip().splitlines()
    assert len(lineas) == 1
    assert json.loads(lineas[0])["estado"] == "completado"

    reanudado = RegistroEstudio.reanudar(
        ruta, familia="prueba", metrica_objetivo="mase", ventanas=_ventanas()
    )
    assert reanudado.trials[0].id == "t1"
    assert reanudado.trials[0].valor == 0.5


def test_reanudar_sin_archivo_devuelve_registro_vacio(tmp_path: Path):
    registro = RegistroEstudio.reanudar(
        tmp_path / "no_existe.jsonl",
        familia="prueba",
        metrica_objetivo="mase",
        ventanas=_ventanas(),
    )
    assert registro.trials == ()


def test_instantanea_elige_el_menor_valor():
    registro = RegistroEstudio(
        familia="prueba", metrica_objetivo="mase", ventanas=_ventanas()
    )
    registro.abrir("t1", {})
    registro.completar("t1", valor=2.0, n_ventanas=10)
    registro.abrir("t2", {})
    registro.completar("t2", valor=0.5, n_ventanas=10)

    instantanea = registro.instantanea()
    assert instantanea.mejor is not None
    assert instantanea.mejor.id == "t2"
    assert instantanea.n_completados == 2


def test_instantanea_mejor_es_none_sin_completados():
    registro = RegistroEstudio(
        familia="prueba", metrica_objetivo="mase", ventanas=_ventanas()
    )
    registro.abrir("t1", {})
    registro.fallar("t1", motivo="x")
    instantanea = registro.instantanea()
    assert instantanea.mejor is None
    assert instantanea.n_fallidos == 1

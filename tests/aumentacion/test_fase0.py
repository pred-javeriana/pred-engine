"""0.4-C1/C2 - Pruebas del orquestador One-Shot y su bitacora."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pred_engine.aumentacion import fase0
from pred_engine.aumentacion.fase0 import (
    ConfiguracionCorrida,
    ejecutar_fase_0,
    main,
)


def _semilla(tmp_path: Path, *, skus: int = 2, n: int = 140) -> Path:
    rng = np.random.default_rng(7)
    filas = []
    for s in range(skus):
        base = np.arange(n)
        demanda = (
            25.0
            + 0.05 * base
            + 6.0 * np.sin(2 * np.pi * base / 7)
            + rng.normal(0.0, 1.5, n)
        ).clip(min=0)
        for i, d in enumerate(demanda):
            filas.append(
                (
                    f"SKU{s}",
                    pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    round(float(d)),
                    3,
                )
            )
    marco = pd.DataFrame(
        filas, columns=["sku_id", "timestamp", "demand_qty", "lead_time_days"]
    )
    destino = tmp_path / "seed.csv"
    marco.to_csv(destino, index=False)
    return destino


def _config(**kwargs) -> ConfiguracionCorrida:
    base = dict(
        period=7,
        n_series_por_sku=4,
        tolerancia_divergencia=0.2,
        max_reintentos=40,
        minimo_filas=0,
    )
    base.update(kwargs)
    return ConfiguracionCorrida(**base)


def test_una_invocacion_produce_el_artefacto_y_la_bitacora(tmp_path: Path) -> None:
    resultado = ejecutar_fase_0(
        _semilla(tmp_path), _config(), data_root=tmp_path / "data"
    )
    assert resultado.artefacto.path.is_file()
    leido = pd.read_csv(resultado.artefacto.path)
    assert list(leido.columns) == [
        "sku_id",
        "timestamp",
        "demand_qty",
        "lead_time_days",
    ]
    assert (leido["demand_qty"] >= 0).all()
    assert (leido["lead_time_days"] >= 1).all()

    assert resultado.bitacora_path.is_file()
    assert resultado.bitacora_path.parent.name == "logs"
    datos = json.loads(resultado.bitacora_path.read_text(encoding="utf-8"))
    assert datos["artefacto_sha256"] == resultado.artefacto.sha256
    assert datos["semilla_aleatoria"] == 42
    assert datos["finalizada_en"] is not None


def test_la_corrida_es_reproducible(tmp_path: Path) -> None:
    semilla = _semilla(tmp_path)
    a = ejecutar_fase_0(semilla, _config(), data_root=tmp_path / "a")
    b = ejecutar_fase_0(semilla, _config(), data_root=tmp_path / "b")
    assert a.artefacto.sha256 == b.artefacto.sha256


def test_la_bitacora_no_se_escribe_en_el_directorio_crudo(tmp_path: Path) -> None:
    ejecutar_fase_0(_semilla(tmp_path), _config(), data_root=tmp_path / "data")
    crudos = list((tmp_path / "data" / "raw").iterdir())
    assert all(p.suffix != ".json" for p in crudos)


def test_semilla_sin_columnas_del_contrato_es_rechazada(tmp_path: Path) -> None:
    mala = tmp_path / "mala.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(mala, index=False)
    with pytest.raises(ValueError, match="columnas"):
        ejecutar_fase_0(mala, _config(), data_root=tmp_path / "data")


def test_skus_demasiado_cortos_se_omiten(tmp_path: Path) -> None:
    corta = tmp_path / "corta.csv"
    pd.DataFrame(
        {
            "sku_id": ["X"] * 5,
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "demand_qty": [1, 2, 3, 4, 5],
            "lead_time_days": [2, 2, 2, 2, 2],
        }
    ).to_csv(corta, index=False)
    with pytest.raises(ValueError, match="longitud suficiente"):
        ejecutar_fase_0(corta, _config(), data_root=tmp_path / "data")


def test_minimo_de_filas_se_aplica_por_defecto(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filas"):
        ejecutar_fase_0(
            _semilla(tmp_path),
            _config(minimo_filas=1_000_000),
            data_root=tmp_path / "data",
        )


def test_cli_ejecuta_la_fase_0(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    semilla = _semilla(tmp_path)
    codigo = main(
        [
            str(semilla),
            "--data-root",
            str(tmp_path / "data"),
            "--n-series",
            "3",
            "--tolerancia",
            "0.2",
            "--max-reintentos",
            "40",
            "--minimo-filas",
            "0",
        ]
    )
    assert codigo == 0
    salida = capsys.readouterr().out
    assert "Artefacto:" in salida and "Bitacora:" in salida


def test_orquestador_no_importa_el_framework_pred() -> None:
    """La frontera arquitectonica prohibe acoplarse al Modulo 1 (ingesta)."""
    arbol = ast.parse(inspect.getsource(fase0))
    modulos: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos.update(alias.name for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.add(nodo.module)
    assert not any("pred_engine.ingesta" in m for m in modulos)
    assert not any("pred_engine.forecasting" in m for m in modulos)

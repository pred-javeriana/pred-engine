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
from pred_engine.aumentacion.errores import WormOverwriteError
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


def _semilla_transaccional(tmp_path: Path) -> Path:
    """Semilla con cabeceras tipo Kaggle y demanda en ~10 % de los dias."""
    rng = np.random.default_rng(11)
    filas = []
    for item in (100, 101, 102):
        dias = np.sort(rng.choice(240, size=24, replace=False))
        for dia in dias:
            filas.append(
                (
                    item,
                    (pd.Timestamp("2024-10-01") + pd.Timedelta(days=int(dia))).date(),
                    int(rng.integers(2, 500)),
                    int(rng.integers(1, 30)),
                )
            )
    marco = pd.DataFrame(
        filas, columns=["Item_ID", "Date", "Avg_Usage_Per_Day", "Restock_Lead_Time"]
    )
    destino = tmp_path / "inventory_data.csv"
    marco.to_csv(destino, index=False)
    return destino


_MAPEO_KAGGLE = (
    ("sku_id", "Item_ID"),
    ("timestamp", "Date"),
    ("demand_qty", "Avg_Usage_Per_Day"),
    ("lead_time_days", "Restock_Lead_Time"),
)


def _directa(**kwargs) -> ConfiguracionCorrida:
    base = dict(
        metodo="mbb-directo",
        n_series_por_sku=3,
        minimo_filas=0,
        mapeo_columnas=_MAPEO_KAGGLE,
    )
    base.update(kwargs)
    return ConfiguracionCorrida(**base)


def test_mbb_directo_conserva_calendario_e_intermitencia(tmp_path: Path) -> None:
    semilla = _semilla_transaccional(tmp_path)
    resultado = ejecutar_fase_0(semilla, _directa(), data_root=tmp_path / "data")

    panel = pd.read_csv(resultado.artefacto.path, parse_dates=["timestamp"])
    original = pd.read_csv(semilla, parse_dates=["Date"])
    assert sorted(panel["sku_id"].unique()) == [
        "100",
        "100::syn000",
        "100::syn001",
        "100::syn002",
        "101",
        "101::syn000",
        "101::syn001",
        "101::syn002",
        "102",
        "102::syn000",
        "102::syn001",
        "102::syn002",
    ]
    for item, grupo in original.groupby("Item_ID"):
        dias = (grupo["Date"].max() - grupo["Date"].min()).days + 1
        media_semilla = grupo["Avg_Usage_Per_Day"].sum() / dias
        for sku in [str(item)] + [f"{item}::syn{i:03d}" for i in range(3)]:
            serie = panel[panel["sku_id"] == sku].sort_values("timestamp")
            # Calendario diario continuo, del primer al ultimo dia de la semilla.
            assert len(serie) == dias
            assert serie["timestamp"].iloc[0] == grupo["Date"].min()
            assert serie["timestamp"].diff().iloc[1:].eq(pd.Timedelta(days=1)).all()
            # 24 dias activos de ~240: la serie sigue siendo intermitente.
            assert 0.8 <= (serie["demand_qty"] == 0).mean() <= 0.95
            assert abs(serie["demand_qty"].mean() / media_semilla - 1) <= 0.05


def test_la_compuerta_mide_la_serie_ya_acotada(tmp_path: Path) -> None:
    resultado = ejecutar_fase_0(
        _semilla(tmp_path), _config(), data_root=tmp_path / "data"
    )
    panel = pd.read_csv(resultado.artefacto.path)
    semilla = panel[panel["sku_id"] == "SKU0"]["demand_qty"]
    for indice in range(4):
        sintetica = panel[panel["sku_id"] == f"SKU0::syn{indice:03d}"]["demand_qty"]
        assert abs(sintetica.mean() / semilla.mean() - 1) <= 0.2
        assert abs(sintetica.var(ddof=0) / semilla.var(ddof=0) - 1) <= 0.2


def test_la_bitacora_registra_metodo_y_mapeo(tmp_path: Path) -> None:
    resultado = ejecutar_fase_0(
        _semilla_transaccional(tmp_path), _directa(), data_root=tmp_path / "data"
    )
    datos = json.loads(resultado.bitacora_path.read_text(encoding="utf-8"))
    assert datos["metodo"] == "mbb-directo"
    assert datos["block_size"] == 30
    assert datos["ruido_relativo"] == 0.03
    assert datos["mapeo_columnas"]["sku_id"] == "Item_ID"
    assert datos["iniciada_en"] <= datos["finalizada_en"]


def test_el_artefacto_queda_de_solo_lectura(tmp_path: Path) -> None:
    resultado = ejecutar_fase_0(
        _semilla(tmp_path), _config(), data_root=tmp_path / "data"
    )
    assert resultado.artefacto.path.stat().st_mode & 0o777 == 0o444


def test_reutilizar_solo_acepta_la_misma_corrida(tmp_path: Path) -> None:
    semilla = _semilla_transaccional(tmp_path)
    datos = tmp_path / "data"
    primera = ejecutar_fase_0(semilla, _directa(), data_root=datos)
    segunda = ejecutar_fase_0(semilla, _directa(), data_root=datos, reutilizar=True)
    assert segunda.reutilizada is True
    assert segunda.artefacto.sha256 == primera.artefacto.sha256
    assert segunda.bitacora_path == primera.bitacora_path
    with pytest.raises(WormOverwriteError):
        ejecutar_fase_0(
            semilla, _directa(semilla_aleatoria=7), data_root=datos, reutilizar=True
        )


def test_mapeo_a_columna_inexistente_es_rechazado(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Item_Code"):
        ejecutar_fase_0(
            _semilla_transaccional(tmp_path),
            _directa(mapeo_columnas=(("sku_id", "Item_Code"),)),
            data_root=tmp_path / "data",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metodo": "gan"},
        {"block_size": 0},
        {"ruido_relativo": 1.0},
        {"mapeo_columnas": (("costo", "Unit_Cost"),)},
        {"mapeo_columnas": (("sku_id", "A"), ("sku_id", "B"))},
    ],
)
def test_configuracion_invalida_falla_antes_de_leer(kwargs) -> None:
    with pytest.raises(ValueError):
        ConfiguracionCorrida(**kwargs)


def test_cli_mapea_columnas_y_elige_metodo(tmp_path, capsys) -> None:
    codigo = main(
        [
            str(_semilla_transaccional(tmp_path)),
            "--data-root",
            str(tmp_path / "data"),
            "--metodo",
            "mbb-directo",
            "--n-series",
            "2",
            "--minimo-filas",
            "0",
            "--columna",
            "sku_id=Item_ID",
            "--columna",
            "timestamp=Date",
            "--columna",
            "demand_qty=Avg_Usage_Per_Day",
            "--columna",
            "lead_time_days=Restock_Lead_Time",
        ]
    )
    assert codigo == 0
    panel = pd.read_csv(tmp_path / "data" / "raw" / "panel_sintetico_fase0.csv")
    assert panel["sku_id"].nunique() == 9


@pytest.mark.parametrize(
    ("argumentos", "mensaje"),
    [
        (["no-existe.csv"], "no existe la semilla"),
        (["{semilla}", "--tolerancia", "0.0001", "--max-reintentos", "1"], "agotado"),
    ],
)
def test_cli_reporta_errores_sin_traza(tmp_path, capsys, argumentos, mensaje) -> None:
    semilla = _semilla(tmp_path)
    argv = [a.replace("{semilla}", str(semilla)) for a in argumentos]
    codigo = main([*argv, "--data-root", str(tmp_path / "data"), "--minimo-filas", "0"])
    error = capsys.readouterr().err
    assert codigo == 1
    assert mensaje in error
    assert "Traceback" not in error

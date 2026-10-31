"""Publicacion 3.5-A1: Parquet derivado de SQLite y vista para M4 (ADR-03-009/010)."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from tests._evaluaciones import (
    T_ESTRELLA,
    metricas_candidato,
    par_fieller_disjunto,
    pronostico,
    serie,
)
from tests._evidencia import (
    base,
    corrida,
    corrida_evaluada,
    fallo,
    filas,
    observaciones,
)

from pred_engine.forecasting.evaluaciones.calculo_errores import FAMILIA_LINEA_BASE
from pred_engine.forecasting.evaluaciones.confirmacion import (
    CategoriaAConfirmar,
    ComparacionFamilia,
    ConfirmacionCategoria,
    Intervalo,
    confirmar_campeones,
)
from pred_engine.forecasting.evaluaciones.diebold_mariano import (
    PruebaDM,
    intervalo_razon,
)
from pred_engine.forecasting.evaluaciones.veredictos import POLITICA_INICIAL
from pred_engine.forecasting.persistencia_evidencia import (
    TABLAS,
    EstadoCorridaError,
    EvidenciaInconsistenteError,
    consultar,
    estado_corrida,
    guardar_evaluacion,
    guardar_unidades,
    publicar,
    repositorio,
)


def _seleccionada(conn, *fallos, prueba_s1: PruebaDM | None = None):
    run_id, evaluaciones, resultado = corrida_evaluada(conn, *fallos)
    if prueba_s1 is not None:
        skus = tuple(
            replace(v, diebold_mariano=prueba_s1) if v.sku == "S1" else v
            for v in resultado.skus
        )
        resultado = replace(resultado, skus=skus)
    guardar_evaluacion(
        conn, run_id, evaluaciones, resultado, observaciones=observaciones(conn, run_id)
    )
    return run_id


def test_publicar_exige_la_corrida_seleccionada(tmp_path: Path) -> None:
    conn = base()
    run_id = corrida(conn)
    with pytest.raises(EstadoCorridaError, match="EN_CURSO"):
        publicar(conn, run_id, tmp_path)
    assert not any(tmp_path.iterdir())


def test_publicar_rechaza_si_un_sku_del_manifiesto_no_tiene_veredicto(
    tmp_path: Path,
) -> None:
    conn = base()
    run_id = _seleccionada(conn, fallo("S9"))
    with pytest.raises(EvidenciaInconsistenteError, match="S9"):
        publicar(conn, run_id, tmp_path)
    assert estado_corrida(conn, run_id) == "SELECCIONADA"
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("filas_por_bloque", [1, 4, 10_000])
def test_publicar_deriva_el_parquet_de_sqlite_y_abre_la_vista(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filas_por_bloque: int
) -> None:
    monkeypatch.setattr(repositorio, "_FILAS_POR_BLOQUE_EXPORTACION", filas_por_bloque)
    conn = base()
    run_id = _seleccionada(
        conn, replace(fallo(), mensaje='Fallo, "señal"\nsegunda línea')
    )
    conn.execute("UPDATE m3_unidades SET semilla = ? WHERE rowid = 3", (2**53 + 1,))
    conn.commit()
    assert filas(conn, "evidencia_publicada") == 0

    rutas = publicar(conn, run_id, tmp_path)

    assert estado_corrida(conn, run_id) == "PUBLICADA"
    assert set(rutas) == set(TABLAS)
    evidencia = consultar(conn, run_id)
    for tabla, ruta in rutas.items():
        assert ruta.parent == tmp_path / run_id
        pd.testing.assert_frame_equal(pd.read_parquet(ruta), evidencia[tabla])
        assert ruta.with_suffix(".csv").read_bytes() == evidencia[tabla].to_csv(
            index=False
        ).encode("utf-8")
    assert [p.name for p in tmp_path.iterdir()] == [run_id]
    # El Parquet de la corrida ya dice PUBLICADA y trae el origen de los datos.
    corrida_parquet = pd.read_parquet(rutas["m3_corridas"])
    assert corrida_parquet["estado"].tolist() == ["PUBLICADA"]
    # Una columna toda NULL conserva el tipo declarado en el esquema.
    veredictos = pd.read_parquet(rutas["m3_veredictos"])
    assert veredictos["dm_p_valor"].isna().all()
    assert veredictos["dm_p_valor"].dtype == "Float64"
    # M4 solo ve corridas publicadas, con el origen de los datos.
    vista = pd.read_sql_query(
        "SELECT sku, datos_sinteticos FROM evidencia_publicada", conn
    )
    assert vista["sku"].tolist() == ["S1", "S2"]
    assert vista["datos_sinteticos"].tolist() == [0, 0]


def test_un_intervalo_de_r_con_dos_tramos_llega_entero_a_sqlite_y_parquet(
    tmp_path: Path,
) -> None:
    intervalo = intervalo_razon(*par_fieller_disjunto(), nivel=0.95)
    fcr = intervalo_razon(*par_fieller_disjunto(), nivel=0.96)
    assert intervalo is not None and fcr is not None
    assert intervalo.inferior_cola is not None and fcr.inferior_cola is not None
    assert not intervalo.contiene(1.0) and not fcr.contiene(1.0)
    conn = base()
    run_id, evaluaciones, resultado = corrida_evaluada(conn)
    skus = tuple(
        replace(v, intervalo_r=intervalo, intervalo_r_fcr=fcr) if v.sku == "S1" else v
        for v in resultado.skus
    )
    guardar_evaluacion(
        conn,
        run_id,
        evaluaciones,
        replace(resultado, skus=skus),
        observaciones=observaciones(conn, run_id),
    )
    rutas = publicar(conn, run_id, tmp_path)

    columnas = [
        "nivel",
        "inferior",
        "superior",
        "inferior_cola",
        "nivel_fcr",
        "inferior_fcr",
        "superior_fcr",
        "inferior_cola_fcr",
    ]
    esperado = [
        0.95,
        intervalo.inferior,
        intervalo.superior,
        intervalo.inferior_cola,
        0.96,
        fcr.inferior,
        fcr.superior,
        fcr.inferior_cola,
    ]
    for publicada in (
        consultar(conn, run_id, sku="S1")["m3_intervalos_r"],
        pd.read_parquet(rutas["m3_intervalos_r"]),
    ):
        assert publicada.loc[publicada["sku"] == "S1", columnas].iloc[0].tolist() == (
            esperado
        )


def _a_confirmar(
    sku_class: str, rmse: dict[str, Sequence[float]]
) -> CategoriaAConfirmar:
    """Doce SKU con RMSE constante por ventana; la campeona es la fundacional."""
    return CategoriaAConfirmar(
        sku_class,  # type: ignore[arg-type]
        "foundation",
        tuple(f for f in rmse if f not in ("foundation", FAMILIA_LINEA_BASE)),
        {
            f"S{i}": {
                f: metricas_candidato(f, 0.8, rmses=[v[i]] * 40)
                for f, v in rmse.items()
            }
            for i in range(12)
        },
    )


def _sin_nulos(valores: Sequence[object]) -> list[object]:
    return [None if v is None or v is pd.NA or v != v else v for v in valores]


def _intervalo(i: Intervalo | None) -> list[float | None]:
    return [None, None] if i is None else [i.inferior, i.superior]


def _fila_comparacion(c: ComparacionFamilia) -> list[object]:
    return [
        c.n_skus,
        c.mediana_razon,
        *_intervalo(c.ic_mediana),
        c.hodges_lehmann,
        *_intervalo(c.ic_hodges_lehmann),
        c.n_victorias,
        *_intervalo(c.ic_victorias),
        c.estadistico_wilcoxon,
        c.p_wilcoxon,
        c.p_iut,
        c.n_mejores,
        c.n_peores,
        c.causa,
    ]


def _fila_confirmacion(c: ConfirmacionCategoria) -> list[object]:
    efecto = c.efecto_linea_base
    assert efecto is not None
    return [
        c.estado,
        c.causa,
        c.p_iut,
        c.p_holm,
        efecto.n_skus,
        efecto.mediana_r,
        *_intervalo(efecto.ic_mediana),
        efecto.causa,
    ]


def test_la_confirmacion_con_perdidas_cero_llega_entera_a_sqlite_y_parquet(
    tmp_path: Path,
) -> None:
    sin_error, uno = [0.0] * 12, [1.0] * 12
    smooth, lumpy = confirmar_campeones(
        (
            # El campeon no se equivoca en S0-S10; dl tampoco en S11.
            _a_confirmar(
                "smooth",
                {
                    FAMILIA_LINEA_BASE: [1.25] * 12,
                    "foundation": [0.0] * 11 + [1.0],
                    "ml": [1.2] * 12,
                    "dl": [1.2] * 11 + [0.0],
                },
            ),
            # SARIMA no se equivoca nunca; Seasonal Naive tampoco en S6-S11.
            _a_confirmar(
                "lumpy",
                {
                    FAMILIA_LINEA_BASE: [1.25] * 6 + sin_error[:6],
                    "foundation": sin_error[:6] + uno[:6],
                    "classical": sin_error,
                },
            ),
        ),
        pd.date_range(T_ESTRELLA, periods=40, freq="D"),
        POLITICA_INICIAL.confirmacion,
        datos_sinteticos=False,
    )
    ml, dl = smooth.comparaciones
    (classical,) = lumpy.comparaciones
    assert (ml.mediana_razon, ml.hodges_lehmann, ml.causa) == (0.0, 0.0, None)
    assert dl.hodges_lehmann is None and dl.causa == "perdida_cero:hodges_lehmann"
    assert classical.mediana_razon == math.inf
    assert smooth.efecto_linea_base is not None
    assert smooth.efecto_linea_base.mediana_r == 0.0
    assert lumpy.efecto_linea_base is not None
    assert lumpy.efecto_linea_base.causa == "perdida_cero:mediana_r,ic_mediana"

    conn = base()
    run_id, evaluaciones, resultado = corrida_evaluada(conn)
    categorias = {c.sku_class: c for c in resultado.categorias}
    resultado = replace(
        resultado,
        categorias=tuple(
            replace(c, confirmacion={"smooth": smooth, "lumpy": lumpy}[c.sku_class])
            for c in resultado.categorias
        ),
    )
    assert set(categorias) == {"smooth", "lumpy"}
    guardar_evaluacion(
        conn, run_id, evaluaciones, resultado, observaciones=observaciones(conn, run_id)
    )
    rutas = publicar(conn, run_id, tmp_path)

    columnas_comparacion = [
        "n_skus",
        "mediana_razon",
        "ic_mediana_inferior",
        "ic_mediana_superior",
        "hodges_lehmann",
        "ic_hodges_lehmann_inferior",
        "ic_hodges_lehmann_superior",
        "n_victorias",
        "ic_victorias_inferior",
        "ic_victorias_superior",
        "estadistico_wilcoxon",
        "p_wilcoxon",
        "p_iut",
        "n_mejores",
        "n_peores",
        "causa",
    ]
    columnas_confirmacion = [
        "estado",
        "causa",
        "p_iut",
        "p_holm",
        "efecto_n_skus",
        "efecto_mediana_r",
        "efecto_ic_inferior",
        "efecto_ic_superior",
        "efecto_causa",
    ]
    evidencia = consultar(conn, run_id)
    for comparaciones, confirmaciones in (
        (evidencia["m3_comparaciones"], evidencia["m3_confirmaciones"]),
        (
            pd.read_parquet(rutas["m3_comparaciones"]),
            pd.read_parquet(rutas["m3_confirmaciones"]),
        ),
    ):
        filas_comparacion = comparaciones.set_index(["sku_class", "competidora"])
        for clase, fila in (
            ("smooth", ml),
            ("smooth", dl),
            ("lumpy", classical),
        ):
            publicada = filas_comparacion.loc[(clase, fila.competidora)]
            assert _sin_nulos(publicada[columnas_comparacion].tolist()) == (
                _sin_nulos(_fila_comparacion(fila))
            )
        filas_confirmacion = confirmaciones.set_index("sku_class")
        for confirmacion in (smooth, lumpy):
            publicada = filas_confirmacion.loc[confirmacion.sku_class]
            assert _sin_nulos(publicada[columnas_confirmacion].tolist()) == (
                _sin_nulos(_fila_confirmacion(confirmacion))
            )


def test_m4_lee_diebold_mariano_en_la_vista_publicada(tmp_path: Path) -> None:
    conn = base()
    prueba = PruebaDM(
        35,
        1,
        estadistico=-3.1,
        p_valor=0.004,
        p_ajustado=0.008,
        significativa=True,
        alfa=0.05,
    )
    run_id = _seleccionada(conn, prueba_s1=prueba)
    publicar(conn, run_id, tmp_path)
    vista = pd.read_sql_query(
        "SELECT sku, dm_n_ventanas, dm_horizonte, dm_estadistico, dm_p_valor,"
        " dm_p_ajustado, dm_alfa, dm_significativa, dm_causa"
        " FROM evidencia_publicada ORDER BY sku",
        conn,
    ).set_index("sku")
    s1 = vista.loc["S1"]
    assert s1.iloc[:7].tolist() == [35, 1, -3.1, 0.004, 0.008, 0.05, 1]
    assert pd.isna(s1["dm_causa"])
    assert vista.loc["S2", "dm_causa"] == "pocas_ventanas"
    assert pd.isna(vista.loc["S2", "dm_p_ajustado"])


def test_si_la_exportacion_falla_la_corrida_no_queda_publicada(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = base()
    run_id = _seleccionada(conn)

    def _falla(*_: object) -> None:
        raise OSError("disco lleno")

    monkeypatch.setattr(
        "pred_engine.forecasting.persistencia_evidencia.repositorio._exportar_tabla",
        _falla,
    )
    with pytest.raises(OSError, match="disco lleno"):
        publicar(conn, run_id, tmp_path)
    assert estado_corrida(conn, run_id) == "SELECCIONADA"
    assert filas(conn, "evidencia_publicada") == 0


def test_fallo_tras_un_archivo_preparado_no_expone_un_conjunto_parcial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = base()
    run_id = _seleccionada(conn)
    original = repositorio._exportar_tabla
    llamadas = 0

    def falla_en_segundo(*args):
        nonlocal llamadas
        llamadas += 1
        if llamadas == 2:
            raise OSError("disco lleno despues del primer archivo")
        original(*args)

    monkeypatch.setattr(repositorio, "_exportar_tabla", falla_en_segundo)
    with pytest.raises(OSError, match="primer archivo"):
        publicar(conn, run_id, tmp_path)
    assert estado_corrida(conn, run_id) == "SELECCIONADA"
    assert not (tmp_path / run_id).exists()
    assert filas(conn, "evidencia_publicada") == 0
    monkeypatch.setattr(repositorio, "_exportar_tabla", original)
    rutas = publicar(conn, run_id, tmp_path)
    assert estado_corrida(conn, run_id) == "PUBLICADA"
    assert pd.read_csv(rutas["m3_veredictos"].with_suffix(".csv"))["sku"].tolist() == [
        "S2",
        "S1",
    ]


def test_fallo_dentro_de_una_tabla_revierte_y_permite_reintentar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = base()
    run_id = _seleccionada(conn)
    monkeypatch.setattr(repositorio, "_FILAS_POR_BLOQUE_EXPORTACION", 4)
    original = pd.DataFrame.to_csv

    def falla_en_bloque(frame, destino, **kwargs):
        if Path(destino.name).stem == "m3_pronosticos" and not kwargs["header"]:
            raise OSError("disco lleno dentro de la tabla")
        return original(frame, destino, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", falla_en_bloque)
    with pytest.raises(OSError, match="dentro de la tabla"):
        publicar(conn, run_id, tmp_path)
    assert estado_corrida(conn, run_id) == "SELECCIONADA"
    assert filas(conn, "evidencia_publicada") == 0
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(pd.DataFrame, "to_csv", original)
    rutas = publicar(conn, run_id, tmp_path)
    pd.testing.assert_frame_equal(
        pd.read_parquet(rutas["m3_pronosticos"]),
        consultar(conn, run_id)["m3_pronosticos"],
    )


@pytest.mark.parametrize("interrumpir", [False, True])
def test_visibilidad_y_recuperacion_tras_renombrar_antes_del_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrumpir: bool
) -> None:
    memoria = base()
    run_id = _seleccionada(memoria)
    with sqlite3.connect(tmp_path / "evidencia.db") as conn:
        memoria.backup(conn)
        with sqlite3.connect(tmp_path / "evidencia.db") as lector:
            destino = tmp_path / "exportaciones"
            renombrar = repositorio.os.replace
            escribir_csv = pd.DataFrame.to_csv
            monkeypatch.setattr(repositorio, "_FILAS_POR_BLOQUE_EXPORTACION", 4)

            def comprobar_lector():
                assert estado_corrida(lector, run_id) == "SELECCIONADA"
                assert filas(lector, "evidencia_publicada") == 0

            def observar_bloque(frame, *args, **kwargs):
                comprobar_lector()
                assert not (destino / run_id).exists()
                return escribir_csv(frame, *args, **kwargs)

            def observar_renombrado(origen, final):
                renombrar(origen, final)
                comprobar_lector()
                assert len(list(Path(final).iterdir())) == 2 * len(TABLAS)
                if interrumpir:
                    raise KeyboardInterrupt("caida antes del commit")

            monkeypatch.setattr(pd.DataFrame, "to_csv", observar_bloque)
            monkeypatch.setattr(repositorio.os, "replace", observar_renombrado)
            if interrumpir:
                with pytest.raises(KeyboardInterrupt, match="antes del commit"):
                    publicar(conn, run_id, destino)
                comprobar_lector()
                monkeypatch.setattr(repositorio.os, "replace", renombrar)
                monkeypatch.setattr(pd.DataFrame, "to_csv", escribir_csv)
                publicar(conn, run_id, destino)
            else:
                publicar(conn, run_id, destino)
            assert estado_corrida(lector, run_id) == "PUBLICADA"
            assert filas(lector, "evidencia_publicada") == 2
            for tabla, frame in consultar(conn, run_id).items():
                pd.testing.assert_frame_equal(
                    pd.read_parquet(destino / run_id / f"{tabla}.parquet"), frame
                )


def test_publicar_borra_el_temporal_de_una_caida_abrupta(tmp_path: Path) -> None:
    conn = base()
    run_id = _seleccionada(conn)
    abandonado = tmp_path / f".{run_id}-abc123"
    abandonado.mkdir()
    (abandonado / "m3_metricas.parquet").write_bytes(b"parcial")

    publicar(conn, run_id, tmp_path)

    assert [p.name for p in tmp_path.iterdir()] == [run_id]


def test_publicar_conserva_los_temporales_de_otras_corridas(tmp_path: Path) -> None:
    conn = base()
    run_id = _seleccionada(conn)
    ajeno = tmp_path / ".m3-otra-corrida-abc123"
    ajeno.mkdir()
    parcial = ajeno / "m3_metricas.parquet"
    parcial.write_bytes(b"parcial de otra corrida")

    publicar(conn, run_id, tmp_path)

    assert parcial.read_bytes() == b"parcial de otra corrida"
    assert {p.name for p in tmp_path.iterdir()} == {run_id, ajeno.name}


def test_una_corrida_publicada_no_se_reescribe(tmp_path: Path) -> None:
    conn = base()
    run_id = _seleccionada(conn)
    rutas = publicar(conn, run_id, tmp_path)
    marcas = {tabla: ruta.stat().st_mtime_ns for tabla, ruta in rutas.items()}

    assert publicar(conn, run_id, tmp_path) == rutas
    assert {tabla: ruta.stat().st_mtime_ns for tabla, ruta in rutas.items()} == marcas
    with pytest.raises(EstadoCorridaError, match="PUBLICADA"):
        guardar_unidades(conn, run_id, "S1", serie("S1/x", [pronostico(0, [1.0])]))

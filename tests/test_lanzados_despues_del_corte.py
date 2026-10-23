"""Productos lanzados despues del corte: un solo t* y exclusiones recuperables.

La ingesta excluye cada SKU sin demanda positiva hasta t* y sigue con los demas.
El corte sale del calendario completo, antes de excluir; el informe M1, la
clasificacion, el reingreso por Parquet y L2 usan ese mismo corte.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pred_engine import cli
from pred_engine.comun.llm.errores import LlmProviderError, LlmTimeoutError
from pred_engine.comun.reserva import ReserveCut
from pred_engine.forecasting.control_reanudacion import ControladorEvaluacion
from pred_engine.forecasting.persistencia_evidencia import abrir_base_plataforma
from pred_engine.ingesta import pipeline as ip
from pred_engine.ingesta.informe import (
    classify_parquet_with_report,
    ingest_csv_with_report,
)
from pred_engine.ingesta.validador_formato.errores import SchemaBarrierError
from pred_engine.pipeline import EvaluationSettings, PipelineInput, summarize_run
from pred_engine.pipeline_setup import build_pipeline
from pred_engine.verify import representative_csv, verification_pipeline
from tests._plataforma import crear_base_plataforma

INICIO = date(2026, 1, 1)
LANZADO = "lanzado despues del corte"


def _serie(sku: str, dias: range, positivos: range) -> list[dict[str, Any]]:
    return [
        {
            "sku_id": sku,
            "timestamp": (INICIO + timedelta(days=d)).isoformat(),
            "demand_qty": float(5 + d % 3) if d in positivos else 0.0,
            "lead_time_days": 7,
        }
        for d in dias
    ]


# Quitar L acorta el calendario por el final: t* bajaria del dia 87 al 79 y A,
# con demanda solo en 80-87, quedaria sin historia admisible.
FIN_DEL_CALENDARIO = (
    _serie("A", range(100), range(80, 88))
    + _serie("B", range(100), range(100))
    + _serie("L", range(89, 110), range(89, 110))
)
# Quitar X acorta el calendario por el inicio: t* subiria del dia 95 al 100 y
# X, con demanda desde el dia 96, contradiria el corte publicado.
INICIO_DEL_CALENDARIO = _serie("B", range(21, 121), range(21, 121)) + _serie(
    "X", range(121), range(96, 121)
)


def _csv(ruta: Path, filas: list[dict[str, Any]]) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(ruta, index=False)
    return ruta


def _parquet_1_2(ruta: Path, filas: list[dict[str, Any]]) -> Path:
    marco = pd.DataFrame(filas).assign(
        timestamp=lambda m: pd.to_datetime(m["timestamp"])
    )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    marco.to_parquet(ruta, index=False)
    return ruta


def _informe(salida: str) -> dict[str, Any]:
    linea = next(x for x in salida.splitlines() if x.startswith("informe: "))
    return json.loads(Path(linea.removeprefix("informe: ")).read_text())


def _parquet_publicado(raiz: Path, informe: dict[str, Any]) -> Path:
    return raiz / informe["published"]["parquet_path"]


def _con_demanda_hasta(filas: list[dict[str, Any]], t_star: str) -> set[str]:
    return {f["sku_id"] for f in filas if f["timestamp"] <= t_star and f["demand_qty"]}


def _late_launch_run_csv(ruta: Path) -> Path:
    """Panel de verificacion mas un SKU que extiende el calendario y llega tarde."""
    base = pd.read_csv(representative_csv(ruta))
    fechas = pd.date_range("2024-01-20", "2024-02-08").strftime("%Y-%m-%d")
    late = pd.DataFrame(
        {
            "sku_id": "late",
            "timestamp": fechas,
            "demand_qty": [0.0 if f <= "2024-01-31" else 9.0 for f in fechas],
            "lead_time_days": 3,
        }
    )
    pd.concat([base, late]).to_csv(ruta, index=False)
    return ruta


@pytest.mark.parametrize(
    ("filas", "excluido", "t_star"),
    [
        (FIN_DEL_CALENDARIO, "L", "2026-03-29"),
        (INICIO_DEL_CALENDARIO, "X", "2026-04-06"),
    ],
    ids=["fin", "inicio"],
)
def test_classify_publica_un_solo_corte_anterior_a_las_exclusiones(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    filas: list[dict[str, Any]],
    excluido: str,
    t_star: str,
) -> None:
    raiz = tmp_path / "data"
    csv = _csv(tmp_path / "entrada" / "panel.csv", filas)

    codigo = cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)])

    salida = capsys.readouterr().out
    assert codigo == 0
    assert "excluded_count: 1" in salida
    informe = _informe(salida)
    corte = ReserveCut.of(pd.DataFrame(filas))
    assert informe["reserve"]["t_star"] == corte.t_star.date().isoformat() == t_star
    assert informe["reserve"]["last_observed"] == corte.last_observed.date().isoformat()
    assert informe["excluded_count"] == 1
    assert informe["exclusions"] == [{"sku_id": excluido, "cause": LANZADO}]
    con_demanda = _con_demanda_hasta(filas, t_star)
    assert excluido not in con_demanda
    publicados = set(pd.read_parquet(_parquet_publicado(raiz, informe))["sku_id"])
    assert publicados == {f["sku_id"] for f in filas} - {excluido}
    assert publicados <= con_demanda
    assert {m["sku_id"] for m in informe["topology"]["metrics"]} == publicados

    reingreso = verification_pipeline().ingest(
        PipelineInput(parquet_path=_parquet_publicado(raiz, informe), data_root=raiz)
    )
    assert reingreso.cut() == corte
    assert [e.sku_id for e in reingreso.exclusions] == [excluido]


def test_l2_y_el_resumen_usan_el_corte_publicado_por_m1(tmp_path: Path) -> None:
    raiz = tmp_path / "data"
    csv = _late_launch_run_csv(tmp_path / "source.csv")
    pipeline = verification_pipeline()

    result = pipeline.run(PipelineInput(csv_path=csv, data_root=raiz))

    informe = json.loads(next((raiz / "ingestas").glob("*.json")).read_text())
    assert informe["reserve"]["t_star"] == "2024-01-31"
    assert result.ingestion is not None
    assert ReserveCut.of(result.ingestion.panel).t_star != pd.Timestamp("2024-01-31")
    summary = summarize_run(result)
    assert summary["reserve"]["t_star"] == "2024-01-31"
    assert summary["reserve"]["reserved_days"] == informe["reserve"]["reserved_days"]
    assert summary["excluded_count"] == 1
    assert result.stages[2].state == "completed"
    resumed = summarize_run(
        pipeline.run(
            PipelineInput(parquet_path=result.ingestion.parquet_path, data_root=raiz)
        )
    )
    assert resumed["reserve"] == summary["reserve"]
    assert resumed["exclusions"] == summary["exclusions"]


def test_classify_parquet_deja_informe_y_run_parquet_recupera_exclusiones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raiz = tmp_path / "data"
    filas = pd.read_csv(_late_launch_run_csv(tmp_path / "source.csv"))
    origen = _parquet_1_2(
        tmp_path / "entrada" / "panel.parquet", filas.to_dict("records")
    )

    codigo = cli.main(["classify", "--parquet", str(origen), "--data-root", str(raiz)])

    salida = capsys.readouterr().out
    assert codigo == 0
    assert "excluded_count: 1" in salida
    assert '"sku_id": "late"' in salida
    informe = _informe(salida)
    assert informe["status"] == "accepted"
    assert informe["source_file"]["name"] == "panel.parquet"
    assert informe["deposit"] is None
    assert informe["excluded_count"] == 1
    assert informe["exclusions"] == [{"sku_id": "late", "cause": LANZADO}]
    assert informe["reserve"]["t_star"] == "2024-01-31"

    result = verification_pipeline().run(
        PipelineInput(parquet_path=_parquet_publicado(raiz, informe), data_root=raiz)
    )
    summary = summarize_run(result)
    assert summary["excluded_count"] == 1
    assert summary["exclusions"] == [
        {"sku_id": "late", "cause": f"sin historia admisible: {LANZADO}"}
    ]
    assert summary["reserve"]["t_star"] == "2024-01-31"


class _ProveedorAceptado:
    def complete(self, prompt: str, *, temperature: float, timeout: float) -> str:
        return json.dumps(
            {
                "status": "accepted",
                "diagnostic": [
                    {
                        "field": "schema",
                        "severity": "info",
                        "message": "Cabeceras canonicas presentes",
                        "action": None,
                    }
                ],
            }
        )


def test_ingest_imprime_las_exclusiones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "build_llm_provider", lambda *a, **k: _ProveedorAceptado())
    csv = _csv(tmp_path / "panel.csv", FIN_DEL_CALENDARIO)

    codigo = cli.main(
        [
            "ingest",
            "--csv",
            str(csv),
            "--provider",
            "gemini",
            "--api-key",
            "test-key",
            "--data-root",
            str(tmp_path / "data"),
        ]
    )

    salida = capsys.readouterr().out
    assert codigo == 0
    assert "excluded_count: 1" in salida
    assert _informe(salida)["exclusions"] == [{"sku_id": "L", "cause": LANZADO}]


def test_sin_exclusiones_classify_no_imprime_conteo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = _csv(tmp_path / "panel.csv", _serie("B", range(100), range(100)))

    assert cli.main(["classify", "--csv", str(csv), "--data-root", str(tmp_path)]) == 0

    assert "excluded_count" not in capsys.readouterr().out


def test_bytes_iguales_de_otra_ingesta_no_aportan_exclusiones(tmp_path: Path) -> None:
    raiz = tmp_path / "data"
    limpio = _serie("A", range(100), range(100)) + _serie("B", range(100), range(100))
    tardio = _serie("tardio", range(40, 60), range(0))
    pipeline = verification_pipeline()
    a = pipeline.ingest(
        PipelineInput(csv_path=_csv(tmp_path / "limpio.csv", limpio), data_root=raiz)
    )
    b = pipeline.ingest(
        PipelineInput(
            csv_path=_csv(tmp_path / "mixto.csv", limpio + tardio), data_root=raiz
        )
    )
    assert a.parquet_sha256 == b.parquet_sha256
    assert [e.sku_id for e in b.exclusions] == ["tardio"]

    reingreso = pipeline.ingest(
        PipelineInput(parquet_path=a.parquet_path, data_root=raiz)
    )

    assert reingreso.exclusions == ()
    assert reingreso.cut() == ReserveCut.of(pd.DataFrame(limpio))


def test_dos_ingestas_publicadas_en_la_misma_ruta_se_rechazan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raiz = tmp_path / "data"
    limpio = _serie("A", range(100), range(100)) + _serie("B", range(100), range(100))
    tardio = _serie("tardio", range(40, 60), range(0))
    parquet = _parquet_1_2(tmp_path / "a" / "panel.parquet", limpio)
    csv = _csv(tmp_path / "b" / "panel.csv", limpio + tardio)
    assert (
        cli.main(["classify", "--parquet", str(parquet), "--data-root", str(raiz)]) == 0
    )
    assert cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)]) == 0
    publicados = [
        json.loads(r.read_text())["published"]
        for r in (raiz / "ingestas").glob("*.json")
    ]
    assert len({(p["parquet_path"], p["sha256"]) for p in publicados}) == 1

    with pytest.raises(ValueError, match="varias procedencias M1"):
        verification_pipeline().ingest(
            PipelineInput(
                parquet_path=raiz / "processed" / "panel.parquet", data_root=raiz
            )
        )


def test_raiz_por_omision_recupera_exclusiones_sin_crear_directorios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raiz = tmp_path / "data"
    monkeypatch.setenv("PRED_DATA_ROOT", str(raiz))
    pipeline = verification_pipeline()
    csv = _csv(tmp_path / "panel.csv", FIN_DEL_CALENDARIO)
    ingesta = pipeline.ingest(PipelineInput(csv_path=csv))

    reingreso = pipeline.ingest(PipelineInput(parquet_path=ingesta.parquet_path))

    assert [e.sku_id for e in reingreso.exclusions] == ["L"]
    assert (
        reingreso.cut()
        == ingesta.cut()
        == ReserveCut.of(pd.DataFrame(FIN_DEL_CALENDARIO))
    )
    otra = tmp_path / "sin-crear"
    monkeypatch.setenv("PRED_DATA_ROOT", str(otra))
    assert (
        pipeline.ingest(PipelineInput(parquet_path=ingesta.parquet_path)).exclusions
        == ()
    )
    assert not otra.exists()


def test_l2_rechaza_un_panel_fuera_del_corte_de_m1(tmp_path: Path) -> None:
    pipeline = verification_pipeline()
    ingesta = pipeline.ingest(
        PipelineInput(
            csv_path=_csv(tmp_path / "panel.csv", FIN_DEL_CALENDARIO),
            data_root=tmp_path / "data",
        )
    )
    posterior = ingesta.panel.assign(
        timestamp=ingesta.panel["timestamp"] + pd.Timedelta(days=30)
    )

    with pytest.raises(ValueError, match="no cubre el calendario"):
        pipeline.fit(replace(ingesta, panel=posterior))


def test_fuente_renombrada_conserva_corte_y_exclusiones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """S1: Un archivo renombrado conserva el corte y las exclusiones."""
    raiz = tmp_path / "data"
    pipeline = verification_pipeline()
    origen_a = _csv(tmp_path / "primero.csv", INICIO_DEL_CALENDARIO)
    origen_b = _csv(tmp_path / "segundo.csv", INICIO_DEL_CALENDARIO)

    assert cli.main(["classify", "--csv", str(origen_a), "--data-root", str(raiz)]) == 0
    salida_a = capsys.readouterr().out
    assert "excluded_count: 1" in salida_a
    informe_a = _informe(salida_a)
    first_parquet = _parquet_publicado(raiz, informe_a)

    reingreso_antes = pipeline.ingest(
        PipelineInput(parquet_path=first_parquet, data_root=raiz)
    )
    assert [e.sku_id for e in reingreso_antes.exclusions] == ["X"]
    assert reingreso_antes.cut().t_star == pd.Timestamp("2026-04-06")
    ajuste_antes = pipeline.fit(reingreso_antes)
    assert ajuste_antes.reserve.t_star == pd.Timestamp("2026-04-06")
    assert len(ajuste_antes.candidates[0].series) == 75
    assert len(ajuste_antes.candidates[0].series) != 80

    assert cli.main(["classify", "--csv", str(origen_b), "--data-root", str(raiz)]) == 0
    capsys.readouterr()

    reingreso_despues = pipeline.ingest(
        PipelineInput(parquet_path=first_parquet, data_root=raiz)
    )
    assert [e.sku_id for e in reingreso_despues.exclusions] == ["X"]
    assert reingreso_despues.cut().t_star == pd.Timestamp("2026-04-06")
    ajuste_despues = pipeline.fit(reingreso_despues)
    assert ajuste_despues.reserve.t_star == pd.Timestamp("2026-04-06")
    assert len(ajuste_despues.candidates[0].series) == 75
    assert len(ajuste_despues.candidates[0].series) != 80

    resumen = summarize_run(
        pipeline.run(PipelineInput(parquet_path=first_parquet, data_root=raiz))
    )
    assert resumen["excluded_count"] == 1
    assert resumen["reserve"]["t_star"] == "2026-04-06"


@pytest.mark.parametrize(
    "publicador",
    ["ingest_canonical_csv", "run_classify_csv", "run_classify_parquet", "run_ingest"],
)
def test_publicadores_publicos_persisten_corte_y_exclusiones_durables(
    tmp_path: Path, publicador: str
) -> None:
    """S2: Cada publicador persiste corte M1 y exclusiones recuperables."""
    raiz = tmp_path / "data"
    pipeline = verification_pipeline()
    csv = _csv(tmp_path / "start.csv", INICIO_DEL_CALENDARIO)
    pq_origen = _parquet_1_2(tmp_path / "start.parquet", INICIO_DEL_CALENDARIO)

    if publicador == "ingest_canonical_csv":
        resultado = ip.ingest_canonical_csv(csv, data_root=raiz)
        salida_pq = resultado.parquet_path
    elif publicador == "run_classify_csv":
        _, salida_pq = ip.run_classify_csv(csv, data_root=raiz)
    elif publicador == "run_classify_parquet":
        _, salida_pq = ip.run_classify_parquet(pq_origen, data_root=raiz)
    else:
        resultado_ingest = ip.run_ingest(csv, _ProveedorAceptado(), data_root=raiz)
        salida_pq = resultado_ingest.parquet_path

    reingreso = pipeline.ingest(PipelineInput(parquet_path=salida_pq, data_root=raiz))
    assert [e.sku_id for e in reingreso.exclusions] == ["X"]
    assert reingreso.cut().t_star == pd.Timestamp("2026-04-06")
    ajuste = pipeline.fit(reingreso)
    assert ajuste.reserve.t_star == pd.Timestamp("2026-04-06")
    assert len(ajuste.candidates[0].series) == 75

    resumen = summarize_run(
        pipeline.run(PipelineInput(parquet_path=salida_pq, data_root=raiz))
    )
    assert resumen["excluded_count"] == 1
    assert resumen["reserve"]["t_star"] == "2026-04-06"


@pytest.mark.parametrize("orden", ["csv_primero", "parquet_primero"])
def test_publicaciones_limpias_equivalentes_ambos_ordenes(
    tmp_path: Path, orden: str
) -> None:
    """S3: Publicaciones equivalentes conservan la identidad sin conflicto."""
    raiz = tmp_path / "data"
    pipeline = verification_pipeline()
    limpio = _serie("A", range(100), range(100)) + _serie("B", range(100), range(100))
    csv = _csv(tmp_path / "panel.csv", limpio)
    pq_12 = _parquet_1_2(tmp_path / "panel.parquet", limpio)

    if orden == "csv_primero":
        assert (
            cli.main(
                [
                    "classify",
                    "--csv",
                    str(csv),
                    "--data-root",
                    str(raiz),
                    "--datos-sinteticos",
                ]
            )
            == 0
        )
        assert (
            cli.main(
                [
                    "classify",
                    "--parquet",
                    str(pq_12),
                    "--data-root",
                    str(raiz),
                    "--datos-sinteticos",
                ]
            )
            == 0
        )
    else:
        assert (
            cli.main(
                [
                    "classify",
                    "--parquet",
                    str(pq_12),
                    "--data-root",
                    str(raiz),
                    "--datos-sinteticos",
                ]
            )
            == 0
        )
        assert (
            cli.main(
                [
                    "classify",
                    "--csv",
                    str(csv),
                    "--data-root",
                    str(raiz),
                    "--datos-sinteticos",
                ]
            )
            == 0
        )

    publicado = raiz / "processed" / "panel.parquet"
    reingreso = pipeline.ingest(PipelineInput(parquet_path=publicado, data_root=raiz))
    assert reingreso.exclusions == ()
    ajuste = pipeline.fit(reingreso)
    assert len(ajuste.candidates) == 2

    resumen = summarize_run(
        pipeline.run(PipelineInput(parquet_path=publicado, data_root=raiz))
    )
    assert resumen["excluded_count"] == 0


def test_reintento_de_origen_rechazado_no_altera_publicacion_previa(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """S4: Un reintento contradictorio de origen falla sin alterar B."""
    raiz = tmp_path / "data"
    limpio = _serie("A", range(100), range(100)) + _serie("B", range(100), range(100))
    diferente = [dict(f, demand_qty=f["demand_qty"] * 2 + 1) for f in limpio]

    pq_a = _parquet_1_2(tmp_path / "a" / "panel.parquet", limpio)
    pq_b = _parquet_1_2(tmp_path / "b" / "panel.parquet", diferente)

    assert (
        cli.main(
            [
                "classify",
                "--parquet",
                str(pq_a),
                "--data-root",
                str(raiz),
                "--datos-sinteticos",
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "classify",
                "--parquet",
                str(pq_b),
                "--data-root",
                str(raiz),
                "--datos-sinteticos",
            ]
        )
        == 0
    )

    dest = raiz / "processed" / "panel.parquet"
    bytes_b_antes = dest.read_bytes()
    primera_demanda_b = float(pd.read_parquet(dest)["demand_qty"].iloc[0])

    codigo = cli.main(
        [
            "classify",
            "--parquet",
            str(pq_a),
            "--data-root",
            str(raiz),
            "--datos-reales",
        ]
    )
    assert codigo != 0
    assert dest.read_bytes() == bytes_b_antes
    assert float(pd.read_parquet(dest)["demand_qty"].iloc[0]) == primera_demanda_b

    with pytest.raises(ValueError, match="origen"):
        classify_parquet_with_report(pq_a, data_root=raiz, datos_sinteticos=False)
    assert dest.read_bytes() == bytes_b_antes
    assert float(pd.read_parquet(dest)["demand_qty"].iloc[0]) == primera_demanda_b


class _ProveedorTimeout:
    def complete(self, prompt: str, *, temperature: float, timeout: float) -> str:
        raise LlmTimeoutError("timeout simulado")


class _ProveedorFallaGenerica:
    def complete(self, prompt: str, *, temperature: float, timeout: float) -> str:
        raise LlmProviderError("error generico simulado")


def _correr_cli_run(
    ruta_parquet: Path,
    raiz: Path,
    run_id: str,
    capsys: pytest.CaptureFixture[str],
    extra: list[str] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    db_path = raiz / f"{run_id}_cli.db"
    crear_base_plataforma(db_path)
    cmd = [
        "run",
        "--parquet",
        str(ruta_parquet),
        "--data-root",
        str(raiz),
        "--db-path",
        str(db_path),
        "--run-id",
        run_id,
        "--families",
        "ml",
        "--trials",
        "1",
        "--min-train",
        "20",
        "--horizon",
        "2",
        "--step",
        "7",
        "--seed",
        "17",
        "--workers",
        "1",
        "--telemetry-interval",
        "0",
    ]
    if extra:
        cmd.extend(extra)
    codigo = cli.main(cmd)
    salida = capsys.readouterr().out
    resumen = None
    for linea in reversed(salida.splitlines()):
        if linea.startswith("{") and '"stages"' in linea:
            resumen = json.loads(linea)
            break
    return codigo, resumen


def _correr_native_m3(
    ruta_parquet: Path,
    raiz: Path,
    run_id: str,
    *,
    datos_sinteticos: bool | None = True,
) -> Any:
    db_path = raiz / f"{run_id}_m3.db"
    crear_base_plataforma(db_path)
    with abrir_base_plataforma(db_path) as con:
        ctl = ControladorEvaluacion(
            con,
            raiz / "runs" / run_id,
            run_id_m2=run_id,
            codigo_version="head",
            datos_sinteticos=datos_sinteticos,
            data_root=raiz,
        )
        settings = EvaluationSettings(min_train=20, horizon=2, step=7)
        native = build_pipeline(
            settings,
            families=("ml",),
            n_trials=1,
            seed=17,
            workers=1,
            validator=ctl,
        )
        return native.run(
            PipelineInput(
                parquet_path=ruta_parquet,
                data_root=raiz,
                datos_sinteticos=datos_sinteticos,
            )
        )


def test_s1_publicaciones_distintas_con_bytes_iguales_no_bloquean_run_parquet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """S1: Publicaciones distintas con bytes iguales no se bloquean entre si."""
    raiz = tmp_path / "data"
    limpio = _serie("A", range(100), range(100)) + _serie("B", range(100), range(100))
    tardio = _serie("tardio", range(40, 60), range(0))
    csv_limpio = _csv(tmp_path / "entrada" / "clean.csv", limpio)
    csv_mixto = _csv(tmp_path / "entrada" / "collision-late.csv", limpio + tardio)

    assert (
        cli.main(
            [
                "classify",
                "--csv",
                str(csv_limpio),
                "--data-root",
                str(raiz),
                "--datos-sinteticos",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        cli.main(
            [
                "classify",
                "--csv",
                str(csv_mixto),
                "--data-root",
                str(raiz),
                "--datos-sinteticos",
            ]
        )
        == 0
    )
    capsys.readouterr()

    pq_limpio = raiz / "processed" / "clean.parquet"
    pq_mixto = raiz / "processed" / "collision-late.parquet"
    assert pq_limpio.exists() and pq_mixto.exists()
    assert pq_limpio.read_bytes() == pq_mixto.read_bytes()

    cod_limpio, res_limpio = _correr_cli_run(
        pq_limpio, raiz, "s1_cli_clean", capsys, ["--datos-sinteticos"]
    )
    assert cod_limpio == 0
    assert res_limpio is not None
    assert res_limpio["excluded_count"] == 0

    cod_mixto, res_mixto = _correr_cli_run(
        pq_mixto, raiz, "s1_cli_mixto", capsys, ["--datos-sinteticos"]
    )
    assert cod_mixto == 0
    assert res_mixto is not None
    assert res_mixto["excluded_count"] == 1
    assert any("tardio" in str(e) for e in res_mixto["exclusions"])

    m3_limpio = _correr_native_m3(pq_limpio, raiz, "s1_m3_clean")
    assert m3_limpio.stages[-1].state == "completed"

    m3_mixto = _correr_native_m3(pq_mixto, raiz, "s1_m3_mixto")
    assert m3_mixto.stages[-1].state == "completed"


def test_s2_reintento_fallido_conserva_corte_y_exclusiones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """S2: Un reintento fallido conserva el corte aceptado y sus exclusiones."""
    raiz = tmp_path / "data"
    pipeline = verification_pipeline()
    csv = _csv(tmp_path / "start.csv", INICIO_DEL_CALENDARIO)

    res_inicial = ingest_csv_with_report(csv, data_root=raiz, datos_sinteticos=True)
    assert res_inicial.error is None
    assert res_inicial.result is not None
    pq_path = res_inicial.result.parquet_path
    hash_antes = pq_path.read_bytes()

    reingreso_antes = pipeline.ingest(
        PipelineInput(parquet_path=pq_path, data_root=raiz)
    )
    assert [e.sku_id for e in reingreso_antes.exclusions] == ["X"]
    assert reingreso_antes.cut().t_star == pd.Timestamp("2026-04-06")
    ajuste_antes = pipeline.fit(reingreso_antes)
    assert len(ajuste_antes.candidates[0].series) == 75

    reintento_timeout = ingest_csv_with_report(
        csv, data_root=raiz, datos_sinteticos=True, provider=_ProveedorTimeout()
    )
    assert isinstance(reintento_timeout.error, LlmTimeoutError)
    assert reintento_timeout.report.status == "failed"
    assert reintento_timeout.path.read_text()
    assert pq_path.read_bytes() == hash_antes

    reingreso_timeout = pipeline.ingest(
        PipelineInput(parquet_path=pq_path, data_root=raiz)
    )
    assert [e.sku_id for e in reingreso_timeout.exclusions] == ["X"]
    assert reingreso_timeout.cut().t_star == pd.Timestamp("2026-04-06")
    ajuste_timeout = pipeline.fit(reingreso_timeout)
    assert len(ajuste_timeout.candidates[0].series) == 75
    assert len(ajuste_timeout.candidates[0].series) != 80

    codigo, resumen = _correr_cli_run(
        pq_path, raiz, "s2_cli_timeout", capsys, ["--datos-sinteticos"]
    )
    assert codigo == 0
    assert resumen is not None
    assert resumen["excluded_count"] == 1
    assert resumen["reserve"]["t_star"] == "2026-04-06"

    reintento_gen = ingest_csv_with_report(
        csv, data_root=raiz, datos_sinteticos=True, provider=_ProveedorFallaGenerica()
    )
    assert isinstance(reintento_gen.error, LlmProviderError)
    assert pq_path.read_bytes() == hash_antes

    reingreso_gen = pipeline.ingest(PipelineInput(parquet_path=pq_path, data_root=raiz))
    assert [e.sku_id for e in reingreso_gen.exclusions] == ["X"]
    assert reingreso_gen.cut().t_star == pd.Timestamp("2026-04-06")
    ajuste_gen = pipeline.fit(reingreso_gen)
    assert len(ajuste_gen.candidates[0].series) == 75


@pytest.mark.parametrize("orden", ["csv_primero", "parquet_primero"])
def test_s3_origen_desconocido_y_conocido_compatibles_ambos_ordenes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], orden: str
) -> None:
    """S3: Declaraciones compatibles (None y True) se combinan en ambos ordenes."""
    raiz = tmp_path / "data"
    pipeline = verification_pipeline()
    limpio = _serie("A", range(100), range(100)) + _serie("B", range(100), range(100))
    csv = _csv(tmp_path / "panel.csv", limpio)
    pq_12 = _parquet_1_2(tmp_path / "panel.parquet", limpio)

    if orden == "csv_primero":
        assert cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)]) == 0
        capsys.readouterr()
        assert (
            cli.main(
                [
                    "classify",
                    "--parquet",
                    str(pq_12),
                    "--data-root",
                    str(raiz),
                    "--datos-sinteticos",
                ]
            )
            == 0
        )
        capsys.readouterr()
    else:
        assert (
            cli.main(
                [
                    "classify",
                    "--parquet",
                    str(pq_12),
                    "--data-root",
                    str(raiz),
                    "--datos-sinteticos",
                ]
            )
            == 0
        )
        capsys.readouterr()
        assert cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)]) == 0
        capsys.readouterr()

    dest = raiz / "processed" / "panel.parquet"
    reingreso = pipeline.ingest(PipelineInput(parquet_path=dest, data_root=raiz))
    assert reingreso.exclusions == ()
    ajuste = pipeline.fit(reingreso)
    assert len(ajuste.candidates) == 2

    m3 = _correr_native_m3(dest, raiz, f"s3_m3_{orden}", datos_sinteticos=True)
    assert m3.stages[-1].state == "completed"

    codigo, resumen = _correr_cli_run(
        dest, raiz, f"s3_cli_{orden}", capsys, ["--datos-sinteticos"]
    )
    assert codigo == 0
    assert resumen is not None

    raiz_conflicto = tmp_path / f"conflicto_{orden}"
    assert (
        cli.main(
            [
                "classify",
                "--csv",
                str(csv),
                "--data-root",
                str(raiz_conflicto),
                "--datos-sinteticos",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        cli.main(
            [
                "classify",
                "--parquet",
                str(pq_12),
                "--data-root",
                str(raiz_conflicto),
                "--datos-reales",
            ]
        )
        == 0
    )
    capsys.readouterr()
    conflicto_pq = raiz_conflicto / "processed" / "panel.parquet"
    with pytest.raises(ValueError, match="varias procedencias M1"):
        pipeline.ingest(
            PipelineInput(parquet_path=conflicto_pq, data_root=raiz_conflicto)
        )
    cod_conflicto, _ = _correr_cli_run(
        conflicto_pq, raiz_conflicto, f"s3_cli_conflicto_{orden}", capsys
    )
    assert cod_conflicto != 0


def test_s4_campo_invalido_en_producto_excluido_rechaza_parquet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """S4: Un campo invalido en un producto excluido rechaza el Parquet con codigo 3."""
    serie_a = _serie("A", range(100), range(100))
    serie_late1 = [
        {
            "sku_id": "late-short",
            "timestamp": (INICIO + timedelta(days=d)).isoformat(),
            "demand_qty": 0.0 if d < 90 else 5.0,
            "lead_time_days": 0,
        }
        for d in range(100)
    ]
    serie_late2 = [
        {
            "sku_id": "late-z",
            "timestamp": (INICIO + timedelta(days=d)).isoformat(),
            "demand_qty": 0.0 if d < 90 else 10.0,
            "lead_time_days": 0,
        }
        for d in range(100)
    ]
    filas = serie_a + serie_late1 + serie_late2

    pq_origen = _parquet_1_2(tmp_path / "invalido.parquet", filas)
    csv_origen = _csv(tmp_path / "invalido.csv", filas)
    raiz = tmp_path / "data"

    cod_csv = cli.main(["classify", "--csv", str(csv_origen), "--data-root", str(raiz)])
    capsys.readouterr()
    assert cod_csv == 3

    cod_pq = cli.main(
        ["classify", "--parquet", str(pq_origen), "--data-root", str(raiz)]
    )
    capsys.readouterr()
    assert cod_pq == 3
    assert not (raiz / "processed" / "invalido.parquet").exists()

    with pytest.raises(SchemaBarrierError):
        ip.run_classify_parquet(pq_origen, data_root=raiz)

    filas_retained_bad = [
        dict(f, lead_time_days=0 if f["sku_id"] == "A" else 7) for f in filas
    ]
    pq_retained_bad = _parquet_1_2(
        tmp_path / "retained_bad.parquet", filas_retained_bad
    )
    cod_retained = cli.main(
        ["classify", "--parquet", str(pq_retained_bad), "--data-root", str(raiz)]
    )
    capsys.readouterr()
    assert cod_retained == 3

    filas_valid_late = [dict(f, lead_time_days=7) for f in filas]
    pq_valid_late = _parquet_1_2(tmp_path / "valid_late.parquet", filas_valid_late)
    cod_valid = cli.main(
        ["classify", "--parquet", str(pq_valid_late), "--data-root", str(raiz)]
    )
    salida_valid = capsys.readouterr().out
    assert cod_valid == 0
    assert "excluded_count: 2" in salida_valid


def test_i1_corte_sobrevive_reintento_renombre_y_republicacion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """I1: El corte y exclusiones sobreviven reintento, renombre y republicacion."""
    raiz = tmp_path / "data"
    pipeline = verification_pipeline()
    csv_a = _csv(tmp_path / "primero.csv", INICIO_DEL_CALENDARIO)
    csv_renombrado = _csv(tmp_path / "segundo.csv", INICIO_DEL_CALENDARIO)

    assert cli.main(["classify", "--csv", str(csv_a), "--data-root", str(raiz)]) == 0
    capsys.readouterr()
    pub = raiz / "processed" / "primero.parquet"

    reingreso1 = pipeline.ingest(PipelineInput(parquet_path=pub, data_root=raiz))
    assert reingreso1.cut().t_star == pd.Timestamp("2026-04-06")
    assert [e.sku_id for e in reingreso1.exclusions] == ["X"]
    assert len(pipeline.fit(reingreso1).candidates[0].series) == 75

    assert (
        cli.main(["classify", "--csv", str(csv_renombrado), "--data-root", str(raiz)])
        == 0
    )
    capsys.readouterr()
    reingreso2 = pipeline.ingest(PipelineInput(parquet_path=pub, data_root=raiz))
    assert reingreso2.cut().t_star == pd.Timestamp("2026-04-06")
    assert [e.sku_id for e in reingreso2.exclusions] == ["X"]
    assert len(pipeline.fit(reingreso2).candidates[0].series) == 75

    ingest_csv_with_report(csv_a, data_root=raiz, provider=_ProveedorTimeout())
    reingreso3 = pipeline.ingest(PipelineInput(parquet_path=pub, data_root=raiz))
    assert reingreso3.cut().t_star == pd.Timestamp("2026-04-06")
    assert [e.sku_id for e in reingreso3.exclusions] == ["X"]
    assert len(pipeline.fit(reingreso3).candidates[0].series) == 75

    assert cli.main(["classify", "--csv", str(csv_a), "--data-root", str(raiz)]) == 0
    capsys.readouterr()
    reingreso4 = pipeline.ingest(PipelineInput(parquet_path=pub, data_root=raiz))
    assert reingreso4.cut().t_star == pd.Timestamp("2026-04-06")
    assert [e.sku_id for e in reingreso4.exclusions] == ["X"]
    assert len(pipeline.fit(reingreso4).candidates[0].series) == 75


def test_i2_reserva_nunca_entra_al_entrenamiento(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """I2: Los dias de reserva nunca entran al entrenamiento."""
    raiz = tmp_path / "data"
    pipeline = verification_pipeline()
    csv = _csv(tmp_path / "panel.csv", INICIO_DEL_CALENDARIO)
    assert cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)]) == 0
    capsys.readouterr()
    pub = raiz / "processed" / "panel.parquet"

    ingest_csv_with_report(csv, data_root=raiz, provider=_ProveedorTimeout())
    reingreso = pipeline.ingest(PipelineInput(parquet_path=pub, data_root=raiz))
    ajuste = pipeline.fit(reingreso)

    corte_t_star = pd.Timestamp("2026-04-06")
    for cand in ajuste.candidates:
        filas = reingreso.panel[
            (reingreso.panel.sku_id == cand.selection.sku_id)
            & (reingreso.panel.timestamp <= ajuste.reserve.t_star)
        ]
        assert len(cand.series) == len(filas)
        assert len(cand.series) == 75
        assert filas.timestamp.max() <= corte_t_star


def test_i3_filas_invalidas_nunca_publicadas_como_aceptadas(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """I3: Filas invalidas en productos excluidos nunca se publican como aceptadas."""
    raiz = tmp_path / "data"
    serie_ok = _serie("A", range(100), range(100))

    serie_bad_lead = [
        {
            "sku_id": "late",
            "timestamp": (INICIO + timedelta(days=d)).isoformat(),
            "demand_qty": 0.0 if d < 90 else 5.0,
            "lead_time_days": 0,
        }
        for d in range(100)
    ]
    pq1 = _parquet_1_2(tmp_path / "bad_lead.parquet", serie_ok + serie_bad_lead)
    assert cli.main(["classify", "--parquet", str(pq1), "--data-root", str(raiz)]) == 3
    capsys.readouterr()
    assert not (raiz / "processed" / "bad_lead.parquet").exists()

    serie_bad_dem = [
        {
            "sku_id": "late",
            "timestamp": (INICIO + timedelta(days=d)).isoformat(),
            "demand_qty": -1.0 if d == 95 else 0.0,
            "lead_time_days": 7,
        }
        for d in range(100)
    ]
    pq2 = _parquet_1_2(tmp_path / "bad_dem.parquet", serie_ok + serie_bad_dem)
    assert cli.main(["classify", "--parquet", str(pq2), "--data-root", str(raiz)]) == 3
    capsys.readouterr()
    assert not (raiz / "processed" / "bad_dem.parquet").exists()

    assert not any(
        json.loads(p.read_text()).get("status") == "accepted"
        for p in (raiz / "ingestas").glob("*.json")
    )


def test_i4_publicaciones_limpias_equivalentes_no_bloquean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """I4: Publicaciones limpias equivalentes no bloquean un artefacto sin cambios."""
    raiz = tmp_path / "data"
    pipeline = verification_pipeline()
    limpio = _serie("A", range(100), range(100)) + _serie("B", range(100), range(100))
    csv = _csv(tmp_path / "panel.csv", limpio)
    pq = _parquet_1_2(tmp_path / "panel.parquet", limpio)

    assert cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)]) == 0
    capsys.readouterr()
    pub = raiz / "processed" / "panel.parquet"
    re1 = pipeline.ingest(PipelineInput(parquet_path=pub, data_root=raiz))
    assert re1.exclusions == ()

    assert (
        cli.main(
            [
                "classify",
                "--parquet",
                str(pq),
                "--data-root",
                str(raiz),
                "--datos-sinteticos",
            ]
        )
        == 0
    )
    capsys.readouterr()
    re2 = pipeline.ingest(PipelineInput(parquet_path=pub, data_root=raiz))
    assert re2.exclusions == ()

    assert cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)]) == 0
    capsys.readouterr()
    re3 = pipeline.ingest(PipelineInput(parquet_path=pub, data_root=raiz))
    assert re3.exclusions == ()

    cod, res = _correr_cli_run(pub, raiz, "i4_run", capsys, ["--datos-sinteticos"])
    assert cod == 0
    assert res is not None


def test_i5_reintento_no_sobrescribe_otra_salida_aceptada(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """I5: Un reintento no sobrescribe otra salida aceptada ni un informe aceptado."""
    raiz = tmp_path / "data"
    limpio = _serie("A", range(100), range(100)) + _serie("B", range(100), range(100))
    csv_a = _csv(tmp_path / "panel_a.csv", limpio)

    res_a = ingest_csv_with_report(csv_a, data_root=raiz, datos_sinteticos=True)
    assert res_a.error is None and res_a.result is not None
    pq_a = res_a.result.parquet_path
    bytes_a = pq_a.read_bytes()
    informe_a_path = res_a.path
    informe_a_texto = informe_a_path.read_text()

    reintento = ingest_csv_with_report(
        csv_a, data_root=raiz, datos_sinteticos=True, provider=_ProveedorTimeout()
    )
    assert reintento.error is not None
    assert pq_a.read_bytes() == bytes_a
    assert informe_a_path.read_text() == informe_a_texto

    with pytest.raises(ValueError, match="origen"):
        ingest_csv_with_report(csv_a, data_root=raiz, datos_sinteticos=False)
    assert pq_a.read_bytes() == bytes_a
    assert informe_a_path.read_text() == informe_a_texto

    diferente = [dict(f, demand_qty=f["demand_qty"] + 10.0) for f in limpio]
    csv_b = _csv(tmp_path / "panel_b.csv", diferente)
    res_b = ingest_csv_with_report(csv_b, data_root=raiz, datos_sinteticos=True)
    assert res_b.error is None and res_b.result is not None
    assert pq_a.read_bytes() == bytes_a
    assert informe_a_path.read_text() == informe_a_texto

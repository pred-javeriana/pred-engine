"""Productos lanzados despues del corte: un solo t* y exclusiones recuperables.

La ingesta excluye cada SKU sin demanda positiva hasta t* y sigue con los demas.
El corte sale del calendario completo, antes de excluir; el informe M1, la
clasificacion, el reingreso por Parquet y L2 usan ese mismo corte.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pred_engine import cli
from pred_engine.comun.reserva import ReserveCut
from pred_engine.pipeline import PipelineInput, summarize_run
from pred_engine.verify import representative_csv, verification_pipeline

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

"""CLI: diagnostico ok, fail-closed, catalogo y clave ausente en stdout."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from pred_engine import cli
from pred_engine.comun.llm import LlmProviderError
from pred_engine.pipeline import Pipeline
from tests._plataforma import crear_base_plataforma


class FakeLlmProvider:
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


class Http503LlmProvider:
    def complete(self, prompt: str, *, temperature: float, timeout: float) -> str:
        raise LlmProviderError(
            "El proveedor LLM respondio HTTP 503",
            status_code=503,
        )


def test_models_lista_gemini_3_5_flash(capsys: pytest.CaptureFixture[str]) -> None:
    codigo = cli.main(["models", "--provider", "gemini"])
    assert codigo == 0
    salida = capsys.readouterr().out
    assert "gemini-3.5-flash" in salida
    assert "gemini-2.5-flash-lite" in salida
    assert "default" in salida.lower() or "Default" in salida


def test_models_proveedor_invalido() -> None:
    assert cli.main(["models", "--provider", "cohere"]) == 1


def test_ingest_rechaza_modelo_fuera_de_catalogo(tmp_path: Path) -> None:
    csv = tmp_path / "mini.csv"
    csv.write_text("Date,Item_ID\n2024-10-01,1\n", encoding="utf-8")
    codigo = cli.main(
        [
            "ingest",
            "--csv",
            str(csv),
            "--provider",
            "gemini",
            "--api-key",
            "test-key",
            "--model",
            "gemini-ultra-falso",
            "--data-root",
            str(tmp_path / "data"),
        ]
    )
    assert codigo == 1


def test_cli_ingest_con_proveedor_inyectado(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv = tmp_path / "mini.csv"
    csv.write_text(
        "sku_id,timestamp,demand_qty,lead_time_days\n"
        "105,2024-10-01,108,17\n"
        "105,2024-10-04,50,17\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "build_llm_provider",
        lambda *args, **kwargs: FakeLlmProvider(),
    )
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
    assert codigo == 0
    salida = capsys.readouterr().out
    assert "no-se-debe-imprimir" not in salida
    assert "filas_panel_diario" in salida
    assert "diagnostico:" in salida
    assert "sku_class_resumen" in salida


def test_cli_sin_clave_falla(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRED_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    csv = tmp_path / "mini.csv"
    csv.write_text("Date,Item_ID\n2024-10-01,1\n", encoding="utf-8")
    codigo = cli.main(
        ["ingest", "--csv", str(csv), "--data-root", str(tmp_path / "data")]
    )
    assert codigo == 1


def test_cli_error_http_llm_sale_limpio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv = tmp_path / "mini.csv"
    csv.write_text(
        "sku_id,timestamp,demand_qty,lead_time_days\n105,2024-10-01,108,17\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "build_llm_provider",
        lambda *args, **kwargs: Http503LlmProvider(),
    )
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
    assert codigo == 1
    err = capsys.readouterr().err
    assert "503" in err
    assert "Traceback" not in err


def test_cli_classify_csv_sin_llm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = tmp_path / "mini.csv"
    csv.write_text(
        "sku_id,timestamp,demand_qty,lead_time_days,Item_Name\n"
        "105,2024-10-01,108,17,Ventilator\n"
        "105,2024-10-04,50,17,Ventilator\n",
        encoding="utf-8",
    )
    codigo = cli.main(
        [
            "classify",
            "--csv",
            str(csv),
            "--data-root",
            str(tmp_path / "data"),
        ]
    )
    assert codigo == 0
    salida = capsys.readouterr().out
    assert "intermittent" in salida
    assert "sku_class_resumen" in salida
    assert "contrato_1_4: accepted" in salida
    assert "Ventilator" not in salida


def test_cli_classify_exige_una_fuente(capsys: pytest.CaptureFixture[str]) -> None:
    codigo = cli.main(["classify", "--data-root", "data"])
    assert codigo == 1
    assert "exactamente uno" in capsys.readouterr().err


def test_cli_verify_parquet_publicado(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = tmp_path / "mini.csv"
    csv.write_text(
        "sku_id,timestamp,demand_qty,lead_time_days\n"
        "105,2024-10-01,108,17\n"
        "105,2024-10-04,50,17\n",
        encoding="utf-8",
    )
    raiz = tmp_path / "data"
    assert cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)]) == 0
    parquet = raiz / "processed" / "mini.parquet"
    codigo = cli.main(["verify", "--parquet", str(parquet)])
    assert codigo == 0
    salida = capsys.readouterr().out
    assert "contrato_1_4: accepted" in salida
    assert "skus: 1" in salida


def test_cli_classify_sku_sin_demanda_sale_6(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = tmp_path / "ceros.csv"
    csv.write_text(
        "sku_id,timestamp,demand_qty,lead_time_days\n"
        "105,2024-10-01,0,17\n"
        "105,2024-10-02,0,17\n",
        encoding="utf-8",
    )
    codigo = cli.main(
        ["classify", "--csv", str(csv), "--data-root", str(tmp_path / "data")]
    )
    assert codigo == 6
    assert "sin demanda" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("etiquetas", "codigo_esperado"),
    [(lambda clase: clase, 0), (lambda clase: clase.encode(), 6)],
    ids=["texto", "bytes"],
)
def test_cli_verify_parquet_rechaza_sku_class_binaria(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    etiquetas: Callable[[str], object],
    codigo_esperado: int,
) -> None:
    csv = tmp_path / "mini.csv"
    csv.write_text(
        "sku_id,timestamp,demand_qty,lead_time_days\n"
        "105,2024-10-01,108,17\n"
        "105,2024-10-04,50,17\n",
        encoding="utf-8",
    )
    raiz = tmp_path / "data"
    assert cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)]) == 0
    panel = pd.read_parquet(raiz / "processed" / "mini.parquet")
    panel["sku_class"] = pd.Series(
        [etiquetas(str(clase)) for clase in panel["sku_class"]], dtype=object
    )
    externo = tmp_path / "externo.parquet"
    panel.to_parquet(externo)
    capsys.readouterr()

    codigo = cli.main(["verify", "--parquet", str(externo)])

    assert codigo == codigo_esperado
    if codigo_esperado:
        assert "violacion Pydantic en fila 0" in capsys.readouterr().err


_FIXTURE_SEMILLA = Path(__file__).parent / "fixtures" / "semilla_inventario.csv"


def _args_run_base(raiz: Path, db: Path, run_id: str) -> list[str]:
    return [
        "run",
        "--seed-csv",
        str(_FIXTURE_SEMILLA),
        "--data-root",
        str(raiz),
        "--db-path",
        str(db),
        "--run-id",
        run_id,
        "--families",
        "classical",
        "--trials",
        "1",
        "--min-train",
        "40",
        "--horizon",
        "2",
        "--step",
        "14",
        "--metric",
        "mae",
        "--seed",
        "17",
        "--workers",
        "2",
        "--telemetry-interval",
        "0",
        "--m0-metodo",
        "mbb-directo",
        "--m0-n-series",
        "1",
        "--m0-minimo-filas",
        "0",
        "--m0-columna",
        "sku_id=Item_ID",
        "--m0-columna",
        "timestamp=Date",
        "--m0-columna",
        "demand_qty=Avg_Usage_Per_Day",
        "--m0-columna",
        "lead_time_days=Restock_Lead_Time",
    ]


def test_cli_run_graba_ajuste_evaluacion_en_corrida_json(tmp_path: Path) -> None:
    db = crear_base_plataforma(tmp_path / "pred.db")
    raiz = tmp_path / "data"
    run_id = "test-eval-settings"
    args = _args_run_base(raiz, db, run_id)

    codigo = cli.main(args)

    assert codigo in (0, 8)
    corrida_path = raiz / "runs" / run_id / "corrida.json"
    assert corrida_path.exists()
    corrida = json.loads(corrida_path.read_text(encoding="utf-8"))
    assert corrida["evaluation"] == {"aggregation": "media_recortada", "trim": 0.1}


def test_cli_run_reanudacion_tras_interrupcion_conserva_identidad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = crear_base_plataforma(tmp_path / "pred.db")
    raiz = tmp_path / "data"
    run_id = "test-resume-identity"
    args = _args_run_base(raiz, db, run_id)

    def interrumpir(*_a: object, **_kw: object) -> None:
        raise KeyboardInterrupt("interrupcion simulada en primera ejecucion")

    monkeypatch.setattr(cli, "_run_pipeline", interrumpir)
    with pytest.raises(KeyboardInterrupt):
        cli.main(args)

    run_dir = raiz / "runs" / run_id
    identidad_path = run_dir / "identidad.json"
    assert identidad_path.exists()
    identidad_previa = json.loads(identidad_path.read_text(encoding="utf-8"))
    assert "evaluation" not in identidad_previa["settings"]

    monkeypatch.undo()
    codigo = cli.main(args)

    assert codigo in (0, 8)
    identidad_reanudada = json.loads(identidad_path.read_text(encoding="utf-8"))
    assert identidad_reanudada == identidad_previa
    corrida = json.loads((run_dir / "corrida.json").read_text(encoding="utf-8"))
    assert corrida["evaluation"] == {"aggregation": "media_recortada", "trim": 0.1}


def test_cli_run_falla_l2_graba_ajuste_evaluacion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = crear_base_plataforma(tmp_path / "pred.db")
    raiz = tmp_path / "data"
    run_id = "test-l2-failure"
    args = _args_run_base(raiz, db, run_id)

    def fallar_fit(*_a: object, **_kw: object) -> None:
        raise RuntimeError("fallo provocado en la seleccion de modelos L2")

    monkeypatch.setattr(Pipeline, "fit", fallar_fit)
    codigo = cli.main(args)

    assert codigo == 1
    corrida_path = raiz / "runs" / run_id / "corrida.json"
    assert corrida_path.exists()
    corrida = json.loads(corrida_path.read_text(encoding="utf-8"))
    assert corrida["evaluation"] == {"aggregation": "media_recortada", "trim": 0.1}
    assert corrida["summary"]["complete"] is False
    etapas = {s["id"]: s["state"] for s in corrida["summary"]["stages"]}
    assert etapas["L2"] == "failed"

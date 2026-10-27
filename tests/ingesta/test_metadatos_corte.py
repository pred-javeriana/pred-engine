"""Pruebas del almacenamiento de t* y exclusiones en metadatos Parquet.

Cubre los criterios de aceptacion y casos borde del rediseño F4:
1. classify --csv excluye productos tardios y escribe metadatos Parquet.
2. run --parquet, Pipeline.ingest y fit usan metadatos sin recalcular t*.
3. run --csv produce resultados identicos a classify --csv + run --parquet.
4. Parquet sin metadatos pred_engine es rechazado pidiendo reimportar.
5. Casos borde: calendario completo, lead_time invalido, venta en t*.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests._plataforma import crear_base_plataforma

from pred_engine import cli
from pred_engine.comun.errores_handoff import MissingCutoffMetadataError
from pred_engine.comun.exclusiones import SkuExclusion
from pred_engine.comun.metadatos_parquet import (
    ParquetMetadata,
    deserializar_metadatos,
    leer_metadatos_parquet,
    serializar_metadatos,
)
from pred_engine.comun.reserva import ReserveCut
from pred_engine.ingesta.pipeline import run_classify_parquet
from pred_engine.pipeline import PipelineInput
from pred_engine.verify import verification_pipeline


def _generar_csv_con_producto_tardio(path: Path) -> Path:
    """Panel de 100 dias (2024-01-01 a 2024-04-09):

    - SKU-ADMITIDO: demanda positiva todos los dias (80 dias admisibles).
    - SKU-TARDIO: ceros del dia 1 al 80 (t*), demanda > 0 del dia 81 al 100.
    t* = 2024-03-20, dias reservados = 20.
    """
    fechas = pd.date_range("2024-01-01", periods=100)
    filas = []
    for i, fecha in enumerate(fechas):
        filas.append(
            {
                "sku_id": "SKU-ADMITIDO",
                "timestamp": fecha,
                "demand_qty": 20.0 + 0.5 * (i % 20),
                "lead_time_days": 3,
            }
        )
        filas.append(
            {
                "sku_id": "SKU-TARDIO",
                "timestamp": fecha,
                "demand_qty": 0.0 if i < 80 else 15.0,
                "lead_time_days": 3,
            }
        )
    pd.DataFrame(filas).to_csv(path, index=False)
    return path


def test_serializacion_metadatos_roundtrip():
    reserva = ReserveCut(
        t_star=cast(pd.Timestamp, pd.Timestamp("2024-01-28")),
        first_reserved=cast(pd.Timestamp, pd.Timestamp("2024-01-29")),
        last_observed=cast(pd.Timestamp, pd.Timestamp("2024-02-04")),
        reserved_days=7,
        fraction=0.2,
    )
    exclusiones = [SkuExclusion("SKU-X", "lanzado despues del corte")]
    meta = ParquetMetadata(
        t_star=reserva.t_star,
        first_reserved=reserva.first_reserved,
        last_observed=reserva.last_observed,
        reserved_days=reserva.reserved_days,
        fraction=reserva.fraction,
        exclusions=tuple(exclusiones),
    )
    serializado = serializar_metadatos(meta)
    recuperado = deserializar_metadatos(serializado)
    assert recuperado.t_star == meta.t_star
    assert recuperado.reserved_days == 7
    assert recuperado.exclusions == tuple(exclusiones)


def test_parquet_sin_metadatos_rechazado_por_pipeline_e_ingesta_y_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # Archivo Parquet valido con las 5 columnas pero SIN metadatos de pred_engine
    tabla = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "sku_id": ["SKU-1"] * 10,
                "timestamp": pd.date_range("2024-01-01", periods=10),
                "demand_qty": [10.0] * 10,
                "lead_time_days": [2] * 10,
                "sku_class": ["smooth"] * 10,
            }
        )
    )
    ruta_parquet = tmp_path / "sin_metadatos.parquet"
    pq.write_table(tabla, ruta_parquet)

    # 1. Pipeline.ingest
    pipeline = verification_pipeline()
    with pytest.raises(MissingCutoffMetadataError, match="re-import from the CSV"):
        pipeline.ingest(PipelineInput(parquet_path=ruta_parquet))

    # 2. run_classify_parquet
    with pytest.raises(MissingCutoffMetadataError, match="re-import from the CSV"):
        run_classify_parquet(ruta_parquet, data_root=tmp_path / "data")

    # 3. CLI classify --parquet
    codigo_classify = cli.main(
        [
            "classify",
            "--parquet",
            str(ruta_parquet),
            "--data-root",
            str(tmp_path / "data_classify"),
        ]
    )
    assert codigo_classify == 6
    err_classify = capsys.readouterr().err
    assert "re-import from the CSV" in err_classify
    assert not (
        tmp_path / "data_classify" / "processed" / "sin_metadatos.parquet"
    ).exists()

    # 4. CLI run --parquet
    base = crear_base_plataforma(tmp_path / "db.sqlite")
    codigo_run = cli.main(
        [
            "run",
            "--parquet",
            str(ruta_parquet),
            "--db-path",
            str(base),
            "--data-root",
            str(tmp_path / "data_run"),
        ]
    )
    assert codigo_run == 1
    err_run = capsys.readouterr().err
    assert "re-import from the CSV" in err_run
    assert not (tmp_path / "data_run" / "runs").exists()


def test_classify_csv_excluye_producto_tardio_y_publica_metadatos(tmp_path: Path):
    csv = tmp_path / "panel.csv"
    _generar_csv_con_producto_tardio(csv)
    raiz = tmp_path / "data"

    codigo = cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)])
    assert codigo == 0

    parquet = raiz / "processed" / "panel.parquet"
    assert parquet.is_file()

    # Contrato de columnas: exactamente las 5 canonicas
    marco = pd.read_parquet(parquet)
    assert list(marco.columns) == [
        "sku_id",
        "timestamp",
        "demand_qty",
        "lead_time_days",
        "sku_class",
    ]
    # El SKU tardio NO esta en los datos publicados
    skus_publicados = set(marco["sku_id"].unique())
    assert skus_publicados == {"SKU-ADMITIDO"}

    # Metadatos del esquema
    meta = leer_metadatos_parquet(parquet)
    assert meta.t_star == pd.Timestamp("2024-03-20")
    assert meta.reserved_days == 20
    assert len(meta.exclusions) == 1
    assert meta.exclusions[0].sku_id == "SKU-TARDIO"
    assert meta.exclusions[0].cause == "lanzado despues del corte"

    # Informe M1
    informes = list((raiz / "ingestas").glob("*.json"))
    assert len(informes) == 1
    informe_json = json.loads(informes[0].read_text(encoding="utf-8"))
    assert informe_json["status"] == "accepted"
    assert informe_json["excluded_count"] == 1
    assert informe_json["exclusions"] == [
        {"sku_id": "SKU-TARDIO", "cause": "lanzado despues del corte"}
    ]
    assert informe_json["reserve"]["t_star"] == "2024-03-20"
    assert informe_json["reserve"]["reserved_days"] == 20


def test_run_parquet_y_pipeline_usan_metadatos_del_archivo(tmp_path: Path):
    csv = tmp_path / "panel.csv"
    _generar_csv_con_producto_tardio(csv)
    raiz = tmp_path / "data"

    # Paso 1: classify --csv
    assert cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)]) == 0
    parquet = raiz / "processed" / "panel.parquet"

    # Paso 2: Pipeline.ingest y fit
    pipeline = verification_pipeline()
    ingestion = pipeline.ingest(PipelineInput(parquet_path=parquet))
    assert ingestion.reserve is not None
    assert ingestion.reserve.t_star == pd.Timestamp("2024-03-20")
    assert len(ingestion.exclusions) == 1
    assert ingestion.exclusions[0].sku_id == "SKU-TARDIO"

    fitting = pipeline.fit(ingestion)
    assert fitting.reserve.t_star == pd.Timestamp("2024-03-20")
    # L2 registra el producto tardio como 'sin historia admisible'
    assert len(fitting.exclusions) == 1
    assert fitting.exclusions[0].sku_id == "SKU-TARDIO"
    assert (
        fitting.exclusions[0].cause
        == "sin historia admisible: lanzado despues del corte"
    )
    assert fitting.exclusions[0].code == "eligibility.no_admissible_history"

    # Verificacion con run --parquet CLI
    base = crear_base_plataforma(tmp_path / "db.sqlite")
    codigo_run = cli.main(
        [
            "run",
            "--parquet",
            str(parquet),
            "--db-path",
            str(base),
            "--data-root",
            str(raiz),
        ]
    )
    assert codigo_run == 0

    corridas = list((raiz / "runs").glob("*/corrida.json"))
    assert len(corridas) == 1
    corrida_json = json.loads(corridas[0].read_text(encoding="utf-8"))
    resumen_corrida = corrida_json["summary"]
    assert resumen_corrida["excluded_count"] == 1
    assert resumen_corrida["exclusions"][0]["sku_id"] == "SKU-TARDIO"
    assert resumen_corrida["reserve"]["t_star"] == "2024-03-20"

    topologias = list((raiz / "runs").glob("*/topologia.json"))
    assert len(topologias) == 1
    topologia_json = json.loads(topologias[0].read_text(encoding="utf-8"))
    assert topologia_json["t_star"] == "2024-03-20"
    assert topologia_json["excluded_count"] == 1


def test_run_csv_identico_a_classify_csv_y_run_parquet(tmp_path: Path):
    csv = tmp_path / "panel.csv"
    _generar_csv_con_producto_tardio(csv)

    # Corrida A: classify --csv seguido de run --parquet
    raiz_a = tmp_path / "data_a"
    assert cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz_a)]) == 0
    parquet_a = raiz_a / "processed" / "panel.parquet"
    base_a = crear_base_plataforma(tmp_path / "db_a.sqlite")
    assert (
        cli.main(
            [
                "run",
                "--parquet",
                str(parquet_a),
                "--db-path",
                str(base_a),
                "--data-root",
                str(raiz_a),
            ]
        )
        == 0
    )

    # Corrida B: run --csv directo sobre el mismo CSV
    raiz_b = tmp_path / "data_b"
    base_b = crear_base_plataforma(tmp_path / "db_b.sqlite")
    assert (
        cli.main(
            [
                "run",
                "--csv",
                str(csv),
                "--db-path",
                str(base_b),
                "--data-root",
                str(raiz_b),
            ]
        )
        == 0
    )

    # Comparar resultados de ambas corridas
    corrida_a = json.loads(
        list((raiz_a / "runs").glob("*/corrida.json"))[0].read_text("utf-8")
    )
    corrida_b = json.loads(
        list((raiz_b / "runs").glob("*/corrida.json"))[0].read_text("utf-8")
    )
    summary_a = corrida_a["summary"]
    summary_b = corrida_b["summary"]

    assert summary_a["reserve"] == summary_b["reserve"]
    assert summary_a["excluded_count"] == summary_b["excluded_count"]
    assert summary_a["exclusions"] == summary_b["exclusions"]

    topologia_a = json.loads(
        list((raiz_a / "runs").glob("*/topologia.json"))[0].read_text("utf-8")
    )
    topologia_b = json.loads(
        list((raiz_b / "runs").glob("*/topologia.json"))[0].read_text("utf-8")
    )
    assert topologia_a["t_star"] == topologia_b["t_star"]
    assert topologia_a["excluded_count"] == topologia_b["excluded_count"]


def test_producto_tardio_con_fechas_finales_no_altera_corte(tmp_path: Path):
    """Caso borde: SKU-ADMITIDO existe solo dias 1..28.

    SKU-TARDIO existe dias 1..35, con ceros dias 1..28 y venta dias 29..35.
    Si se excluye SKU-TARDIO, el panel resultante solo llega hasta el dia 28.
    Sin embargo, t* DEBE calcularse sobre el calendario completo (dias 1..35)
    antes de la exclusion. Por tanto t* = dia 28, dias reservados = 7.
    """
    fechas_admitido = pd.date_range("2024-01-01", periods=28)
    fechas_tardio = pd.date_range("2024-01-01", periods=35)
    filas = []
    for i, f in enumerate(fechas_admitido):
        filas.append(
            {
                "sku_id": "SKU-ADMITIDO",
                "timestamp": f,
                "demand_qty": 25.0,
                "lead_time_days": 3,
            }
        )
    for i, f in enumerate(fechas_tardio):
        filas.append(
            {
                "sku_id": "SKU-TARDIO",
                "timestamp": f,
                "demand_qty": 0.0 if i < 28 else 10.0,
                "lead_time_days": 3,
            }
        )
    csv = tmp_path / "cola_calendario.csv"
    pd.DataFrame(filas).to_csv(csv, index=False)
    raiz = tmp_path / "data"

    assert cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)]) == 0
    parquet = raiz / "processed" / "cola_calendario.parquet"
    meta = leer_metadatos_parquet(parquet)

    # Corte calculado del calendario COMPLETO (35 dias)
    assert meta.t_star == pd.Timestamp("2024-01-28")
    assert meta.reserved_days == 7
    assert meta.last_observed == pd.Timestamp("2024-02-04")
    assert len(meta.exclusions) == 1
    assert meta.exclusions[0].sku_id == "SKU-TARDIO"


def test_mismos_bytes_csv_o_reintento_independientes(tmp_path: Path):
    """Caso borde: mismos bytes importados con dos nombres diferentes.

    Cada Parquet porta sus propios metadatos sin ninguna consulta de informe compartido.
    """
    csv1 = tmp_path / "origen1.csv"
    csv2 = tmp_path / "origen2.csv"
    _generar_csv_con_producto_tardio(csv1)
    _generar_csv_con_producto_tardio(csv2)

    raiz = tmp_path / "data"
    assert cli.main(["classify", "--csv", str(csv1), "--data-root", str(raiz)]) == 0
    assert cli.main(["classify", "--csv", str(csv2), "--data-root", str(raiz)]) == 0

    p1 = raiz / "processed" / "origen1.parquet"
    p2 = raiz / "processed" / "origen2.parquet"
    assert p1.is_file() and p2.is_file()

    m1 = leer_metadatos_parquet(p1)
    m2 = leer_metadatos_parquet(p2)
    assert m1.t_star == m2.t_star == pd.Timestamp("2024-03-20")
    assert m1.exclusions == m2.exclusions


def test_producto_tardio_con_lead_time_invalido_falla_precondicion(tmp_path: Path):
    """Caso borde: un producto que seria excluido por tardio tiene lead_time_days <= 0.

    Debe fallar con el error de esquema/precondicion (exit 3) y no publicar nada.
    """
    fechas = pd.date_range("2024-01-01", periods=35)
    filas = []
    for i, fecha in enumerate(fechas):
        filas.append(
            {
                "sku_id": "SKU-BUENO",
                "timestamp": fecha,
                "demand_qty": 10.0,
                "lead_time_days": 3,
            }
        )
        filas.append(
            {
                "sku_id": "SKU-TARDIO-INVALIDO",
                "timestamp": fecha,
                "demand_qty": 0.0 if i < 28 else 5.0,
                "lead_time_days": 0,  # Invalido!
            }
        )
    csv = tmp_path / "lead_time_cero.csv"
    pd.DataFrame(filas).to_csv(csv, index=False)
    raiz = tmp_path / "data"

    codigo = cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)])
    assert codigo == 3
    assert not (raiz / "processed" / "lead_time_cero.parquet").exists()


def test_producto_con_primera_demanda_exactamente_en_t_star_se_conserva(tmp_path: Path):
    """Caso borde: la primera venta de un producto ocurre exactamente el dia t*.

    Debe conservarse (no excluirse).
    """
    fechas = pd.date_range("2024-01-01", periods=35)
    filas = []
    for i, fecha in enumerate(fechas):
        filas.append(
            {
                "sku_id": "SKU-REGULAR",
                "timestamp": fecha,
                "demand_qty": 10.0,
                "lead_time_days": 3,
            }
        )
        # SKU-EXACTO tiene 0 los dias 0..26, y venta positiva el dia 27
        filas.append(
            {
                "sku_id": "SKU-EXACTO",
                "timestamp": fecha,
                "demand_qty": 0.0 if i < 27 else 8.0,
                "lead_time_days": 3,
            }
        )
    csv = tmp_path / "exacto_en_corte.csv"
    pd.DataFrame(filas).to_csv(csv, index=False)
    raiz = tmp_path / "data"

    codigo = cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)])
    assert codigo == 0

    parquet = raiz / "processed" / "exacto_en_corte.parquet"
    meta = leer_metadatos_parquet(parquet)
    # Ningun SKU excluido
    assert len(meta.exclusions) == 0
    marco = pd.read_parquet(parquet)
    assert set(marco["sku_id"].unique()) == {"SKU-REGULAR", "SKU-EXACTO"}


def test_todos_los_productos_tardios_rechazan_panel_completo(tmp_path: Path):
    """Caso borde: todos los productos tienen demanda 0 hasta t*.

    Debe fallar con HandoffPreconditionError (exit 6) y no publicar nada.
    """
    fechas = pd.date_range("2024-01-01", periods=35)
    filas = []
    for i, fecha in enumerate(fechas):
        filas.append(
            {
                "sku_id": "SKU-TARDIO-1",
                "timestamp": fecha,
                "demand_qty": 0.0 if i < 28 else 10.0,
                "lead_time_days": 3,
            }
        )
        filas.append(
            {
                "sku_id": "SKU-TARDIO-2",
                "timestamp": fecha,
                "demand_qty": 0.0 if i < 28 else 20.0,
                "lead_time_days": 3,
            }
        )
    csv = tmp_path / "todos_tardios.csv"
    pd.DataFrame(filas).to_csv(csv, index=False)
    raiz = tmp_path / "data"

    codigo = cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)])
    assert codigo == 6
    assert not (raiz / "processed" / "todos_tardios.parquet").exists()


def test_s1_verify_rechaza_parquet_sin_metadatos(tmp_path: Path) -> None:
    """``verify --parquet`` aplica la misma guarda de metadatos que run y classify."""
    ruta_parquet = tmp_path / "sin_metadatos.parquet"
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "sku_id": ["SKU-1"] * 10,
                    "timestamp": pd.date_range("2024-01-01", periods=10),
                    "demand_qty": [10.0] * 10,
                    "lead_time_days": [2] * 10,
                    "sku_class": ["smooth"] * 10,
                }
            )
        ),
        ruta_parquet,
    )
    ejecutable = shutil.which("pred-engine", path=str(Path(sys.executable).parent))
    assert ejecutable is not None

    proceso = subprocess.run(
        [ejecutable, "verify", "--parquet", str(ruta_parquet)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proceso.returncode == 6
    assert "re-import from the CSV" in proceso.stderr
    assert "contrato_1_4: accepted" not in proceso.stdout

"""Pruebas del almacenamiento de t* y exclusiones en metadatos Parquet.

Cubre los criterios de aceptacion y casos borde del rediseño F4:
1. classify --csv excluye productos tardios y escribe metadatos Parquet.
2. run --parquet, Pipeline.ingest y fit usan metadatos sin recalcular t*.
3. run --csv produce resultados identicos a classify --csv + run --parquet.
4. Parquet sin metadatos pred_engine es rechazado pidiendo reimportar.
5. Casos borde: calendario completo, lead_time invalido, venta en t*.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar, cast

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

T = TypeVar("T")


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
            "--no-foundation",
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
            "--no-foundation",
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
                "--no-foundation",
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
                "--no-foundation",
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


def _generar_csv_sin_tardios(path: Path, dias: int) -> Path:
    """Dos productos con demanda positiva todos los dias desde 2024-01-01."""
    filas = []
    for i, fecha in enumerate(pd.date_range("2024-01-01", periods=dias)):
        filas.append(
            {
                "sku_id": "SKU-ADMITIDO",
                "timestamp": fecha,
                "demand_qty": 100.0 + i,
                "lead_time_days": 3,
            }
        )
        filas.append(
            {
                "sku_id": "SKU-OTRO",
                "timestamp": fecha,
                "demand_qty": 15.0 + i % 3,
                "lead_time_days": 3,
            }
        )
    pd.DataFrame(filas).to_csv(path, index=False)
    return path


@contextmanager
def _reemplazar_en_primera_lectura(
    vivo: Path, reemplazo: Path, *, antes: bool
) -> Iterator[list[str]]:
    """Sustituye ``vivo`` por otra publicacion valida en su primera lectura.

    Con ``antes`` la sustitucion ocurre justo antes de esa lectura; si no, justo
    despues, de modo que toda lectura posterior de la ruta ve otra publicacion.
    """
    disparos: list[str] = []
    original_bytes = Path.read_bytes
    original_init = pq.ParquetFile.__init__
    original_pandas = pd.read_parquet

    def es_vivo(fuente: object) -> bool:
        return isinstance(fuente, str | os.PathLike) and (
            Path(fuente).expanduser().resolve() == vivo.resolve()
        )

    def leer(api: str, fuente: object, lectura: Callable[[], T]) -> T:
        if disparos or not es_vivo(fuente):
            return lectura()
        if antes:
            disparos.append(api)
            shutil.copyfile(reemplazo, vivo)
            return lectura()
        valor = lectura()
        disparos.append(api)
        shutil.copyfile(reemplazo, vivo)
        return valor

    def read_bytes(self: Path) -> bytes:
        return leer("Path.read_bytes", self, lambda: original_bytes(self))

    def init(self: pq.ParquetFile, source: Any, *args: Any, **kwargs: Any) -> None:
        leer(
            "ParquetFile",
            source,
            lambda: original_init(self, source, *args, **kwargs),
        )

    def read_parquet(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return leer(
            "read_parquet",
            path,
            lambda: original_pandas(path, *args, **kwargs),
        )

    with pytest.MonkeyPatch.context() as parche:
        parche.setattr(Path, "read_bytes", read_bytes)
        parche.setattr(pq.ParquetFile, "__init__", init)
        parche.setattr(pd, "read_parquet", read_parquet)
        yield disparos


def _demanda(marco: pd.DataFrame, sku_id: str, hasta: pd.Timestamp) -> list[float]:
    fila = marco.loc[(marco["sku_id"] == sku_id) & (marco["timestamp"] <= hasta)]
    return fila.sort_values("timestamp")["demand_qty"].astype(float).tolist()


@pytest.mark.parametrize(
    "ruta", ["biblioteca_parquet", "reclasificacion_parquet", "biblioteca_csv"]
)
def test_s2_lectura_unica_no_mezcla_publicaciones(tmp_path: Path, ruta: str) -> None:
    """Una publicacion sustituida entre lecturas no mezcla filas, huella y corte."""
    # Publicacion A: 100 dias, t*=2024-03-20, SKU-TARDIO excluido.
    csv_a = _generar_csv_con_producto_tardio(tmp_path / "panel.csv")
    assert (
        cli.main(["classify", "--csv", str(csv_a), "--data-root", str(tmp_path / "a")])
        == 0
    )
    # Publicacion B, con el mismo nombre: 125 dias, otro t* y sin exclusiones.
    (tmp_path / "fuente_b").mkdir()
    csv_b = _generar_csv_sin_tardios(tmp_path / "fuente_b" / "panel.csv", 125)
    assert (
        cli.main(["classify", "--csv", str(csv_b), "--data-root", str(tmp_path / "b")])
        == 0
    )
    pub_a = tmp_path / "a" / "processed" / "panel.parquet"
    pub_b = tmp_path / "b" / "processed" / "panel.parquet"
    meta_a, meta_b = leer_metadatos_parquet(pub_a), leer_metadatos_parquet(pub_b)
    assert meta_a.t_star == pd.Timestamp("2024-03-20")
    assert meta_b.t_star == pd.Timestamp("2024-04-09")
    filas_a, filas_b = pd.read_parquet(pub_a), pd.read_parquet(pub_b)
    sha_b = hashlib.sha256(pub_b.read_bytes()).hexdigest()

    if ruta == "reclasificacion_parquet":
        vivo = tmp_path / "vivo" / "panel.parquet"
        vivo.parent.mkdir()
        shutil.copyfile(pub_a, vivo)
        salida = tmp_path / "reclasificado"
        with _reemplazar_en_primera_lectura(vivo, pub_b, antes=False) as disparos:
            assert (
                cli.main(
                    ["classify", "--parquet", str(vivo), "--data-root", str(salida)]
                )
                == 0
            )
        assert disparos
        publicado = salida / "processed" / "panel.parquet"
        # Filas y metadatos de la misma publicacion A, la que se leyo primero.
        pd.testing.assert_frame_equal(pd.read_parquet(publicado), filas_a)
        assert leer_metadatos_parquet(publicado) == meta_a
        return

    pipeline = verification_pipeline()
    if ruta == "biblioteca_parquet":
        # La primera lectura obtiene A; despues la ruta ya contiene B.
        vivo = tmp_path / "vivo" / "panel.parquet"
        vivo.parent.mkdir()
        shutil.copyfile(pub_a, vivo)
        esperado, meta, filas = (
            hashlib.sha256(pub_a.read_bytes()).hexdigest(),
            meta_a,
            filas_a,
        )
        with _reemplazar_en_primera_lectura(vivo, pub_b, antes=False) as disparos:
            ingestion = pipeline.ingest(PipelineInput(parquet_path=vivo))
    else:
        # La importacion publica A; otra publicacion la sustituye antes de releerla.
        datos = tmp_path / "datos_csv"
        vivo = datos / "processed" / "panel.parquet"
        esperado, meta, filas = sha_b, meta_b, filas_b
        with _reemplazar_en_primera_lectura(vivo, pub_b, antes=True) as disparos:
            ingestion = pipeline.ingest(PipelineInput(csv_path=csv_a, data_root=datos))
    assert disparos

    assert ingestion.parquet_sha256 == esperado
    assert ingestion.reserve == meta.reserve
    assert ingestion.exclusions == meta.exclusions
    assert len(ingestion.panel) == len(filas)
    assert sorted(m.sku_id for m in ingestion.topology) == sorted(
        filas["sku_id"].unique()
    )

    fitted = pipeline.fit(ingestion)
    assert fitted.reserve == meta.reserve
    excluidos_l1 = [
        e.sku_id
        for e in fitted.exclusions
        if e.code == "eligibility.no_admissible_history"
    ]
    assert excluidos_l1 == [e.sku_id for e in meta.exclusions]
    entrenamiento = {c.selection.sku_id: c.series.tolist() for c in fitted.candidates}
    assert entrenamiento
    for sku_id, serie in entrenamiento.items():
        # Ninguna observacion reservada de la publicacion leida llega al ajuste.
        assert serie == _demanda(filas, sku_id, meta.t_star)


def test_s1_fit_rechaza_artefacto_sin_corte_y_no_relee_la_ruta(tmp_path: Path) -> None:
    """Un artefacto L1 armado a mano sin corte no toma t* de la ruta al ajustar."""
    from pred_engine.ingesta.salida import read_classified_publication
    from pred_engine.pipeline import IngestionArtifact

    csv_a = _generar_csv_con_producto_tardio(tmp_path / "panel.csv")
    assert (
        cli.main(["classify", "--csv", str(csv_a), "--data-root", str(tmp_path / "a")])
        == 0
    )
    (tmp_path / "fuente_b").mkdir()
    csv_b = _generar_csv_sin_tardios(tmp_path / "fuente_b" / "panel.csv", 125)
    assert (
        cli.main(["classify", "--csv", str(csv_b), "--data-root", str(tmp_path / "b")])
        == 0
    )
    pub_a = tmp_path / "a" / "processed" / "panel.parquet"
    pub_b = tmp_path / "b" / "processed" / "panel.parquet"
    vivo = tmp_path / "vivo" / "panel.parquet"
    vivo.parent.mkdir()
    shutil.copyfile(pub_a, vivo)
    filas_a, meta_a = pd.read_parquet(pub_a), leer_metadatos_parquet(pub_a)

    publicacion = read_classified_publication(vivo)
    a_mano = IngestionArtifact(
        vivo, publicacion.frame, parquet_sha256=publicacion.sha256
    )
    # Otra publicacion valida ocupa la ruta antes de L2.
    shutil.copyfile(pub_b, vivo)

    pipeline = verification_pipeline()
    with pytest.raises(ValueError, match="no declara su corte"):
        pipeline.fit(a_mano)

    # La ruta publica arma el artefacto con la misma lectura de las filas.
    from pred_engine.pipeline import ingestion_artifact_from_publication

    publica = ingestion_artifact_from_publication(vivo, publicacion)
    fitted = pipeline.fit(publica)
    assert fitted.reserve == meta_a.reserve
    assert [e.sku_id for e in fitted.exclusions] == ["SKU-TARDIO"]
    for candidato in fitted.candidates:
        sku_id = candidato.selection.sku_id
        assert candidato.series.tolist() == _demanda(filas_a, sku_id, meta_a.t_star)


def _corrida(raiz: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    (corrida,) = (raiz / "runs").glob("*/corrida.json")
    topologia = corrida.parent / "topologia.json"
    return (
        json.loads(corrida.read_text("utf-8"))["summary"],
        json.loads(topologia.read_text("utf-8")),
    )


def _correr(tmp_path: Path, raiz: Path, *fuente: str) -> int:
    base = crear_base_plataforma(tmp_path / f"{raiz.name}.sqlite")
    return cli.main(
        [
            "run",
            *fuente,
            "--db-path",
            str(base),
            "--data-root",
            str(raiz),
            "--families",
            "ml",
            "--trials",
            "1",
            "--workers",
            "1",
            "--min-train",
            "40",
            "--horizon",
            "2",
            "--step",
            "7",
            "--seed",
            "17",
            "--telemetry-interval",
            "0",
        ]
    )


@pytest.mark.parametrize(
    ("tardio", "esperado"), [(True, 2), (False, 1)], ids=["mixto", "control"]
)
def test_s3_topologia_cuenta_las_mismas_exclusiones_que_el_resumen(
    tmp_path: Path, tardio: bool, esperado: int
) -> None:
    """topologia.json y corrida.json declaran el mismo excluded_count."""
    filas = []
    for i, fecha in enumerate(pd.date_range("2024-01-01", periods=100)):
        filas.append(
            {
                "sku_id": "SKU-ADMITIDO",
                "timestamp": fecha,
                "demand_qty": 20.0 + 0.5 * (i % 20),
                "lead_time_days": 3,
            }
        )
        if tardio:
            filas.append(
                {
                    "sku_id": "SKU-TARDIO",
                    "timestamp": fecha,
                    "demand_qty": 0.0 if i < 80 else 15.0,
                    "lead_time_days": 3,
                }
            )
        if i < 30:
            # Termina antes de t*: L1 lo conserva y L2 lo excluye.
            filas.append(
                {
                    "sku_id": "SKU-CORTO",
                    "timestamp": fecha,
                    "demand_qty": 9.0,
                    "lead_time_days": 3,
                }
            )
    csv = tmp_path / "mixto.csv"
    pd.DataFrame(filas).to_csv(csv, index=False)

    raiz = tmp_path / "data"
    assert _correr(tmp_path, raiz, "--csv", str(csv)) == 0
    resumen, topologia = _corrida(raiz)

    assert resumen["excluded_count"] == esperado
    assert topologia["excluded_count"] == resumen["excluded_count"]
    assert topologia["t_star"] == resumen["reserve"]["t_star"] == "2024-03-20"


@pytest.mark.parametrize("fuente", ["csv", "parquet"])
def test_s4_resumen_de_l2_fallida_conserva_su_reserva(
    tmp_path: Path, fuente: str
) -> None:
    """Una corrida que falla en L2 guarda el corte que L1 ya conocia."""
    filas = []
    for i, fecha in enumerate(pd.date_range("2024-01-01", periods=35)):
        filas.append(
            {
                "sku_id": "SKU-CORTO",
                "timestamp": fecha,
                "demand_qty": 10.0 + i % 4,
                "lead_time_days": 3,
            }
        )
        filas.append(
            {
                "sku_id": "SKU-TARDIO",
                "timestamp": fecha,
                "demand_qty": 0.0 if i < 28 else 6.0,
                "lead_time_days": 3,
            }
        )
    csv = tmp_path / "corto.csv"
    pd.DataFrame(filas).to_csv(csv, index=False)
    raiz = tmp_path / "data"
    if fuente == "parquet":
        assert cli.main(["classify", "--csv", str(csv), "--data-root", str(raiz)]) == 0
        entrada = ["--parquet", str(raiz / "processed" / "corto.parquet")]
    else:
        entrada = ["--csv", str(csv)]

    assert _correr(tmp_path, raiz, *entrada) == 1
    resumen, topologia = _corrida(raiz)

    assert resumen["complete"] is False
    assert [e["state"] for e in resumen["stages"]] == [
        "completed",
        "failed",
        "pending",
        "pending",
    ]
    assert resumen["reserve"] == {
        "fraction": 0.2,
        "t_star": "2024-01-28",
        "first_reserved": "2024-01-29",
        "last_observed": "2024-02-04",
        "reserved_days": 7,
    }
    assert resumen["exclusions"] == [
        {
            "sku_id": "SKU-TARDIO",
            "cause": "lanzado despues del corte",
            "code": None,
        }
    ]
    assert topologia["t_star"] == resumen["reserve"]["t_star"]

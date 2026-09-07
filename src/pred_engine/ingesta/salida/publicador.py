"""Publicacion del artefacto Parquet 1.4 hacia processed/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pred_engine.comun.logger import get_logger
from pred_engine.ingesta.lector import export_parquet
from pred_engine.ingesta.salida.errores import OutputContractError
from pred_engine.ingesta.salida.preservacion import require_panel_preserved
from pred_engine.ingesta.salida.validador import (
    normalize_output_frame,
    validate_output_contract,
)

_logger = get_logger(__name__)


def publish_classified_panel(
    classified: pd.DataFrame,
    destination: str | Path,
    *,
    data_root: str | Path | None = None,
    daily_panel: pd.DataFrame,
) -> Path:
    """Valida preservacion + contrato y escribe Parquet en processed/.

    No importa ADI/CV² ni el enrutador SBC. No escribe CSV ni toca raw/.
    """
    destino = Path(destination)
    if destino.suffix.lower() != ".parquet":
        raise OutputContractError("el artefacto final solo se publica como .parquet")

    require_panel_preserved(daily_panel, classified)
    validate_output_contract(classified)
    normalizado = normalize_output_frame(classified)
    escrito = export_parquet(normalizado, destino, data_root=data_root)
    _logger.info(
        "Artefacto 1.4 publicado ruta=%s filas=%s skus=%s",
        escrito,
        int(len(normalizado)),
        len({str(v) for v in normalizado["sku_id"].tolist()}),
    )
    return escrito


def read_classified_parquet(path: str | Path) -> pd.DataFrame:
    """Lee un Parquet y vuelve a imponer el contrato 1.4. No escribe."""
    origen = Path(path).expanduser().resolve()
    if not origen.is_file():
        raise FileNotFoundError(origen)
    if origen.suffix.lower() != ".parquet":
        raise OutputContractError("solo se verifica un artefacto .parquet")
    marco = pd.read_parquet(origen, engine="pyarrow")
    validate_output_contract(marco)
    _logger.info(
        "Artefacto 1.4 verificado ruta=%s filas=%s skus=%s",
        origen,
        int(len(marco)),
        len({str(v) for v in marco["sku_id"].tolist()}),
    )
    return marco

"""Almacen de manifiestos en sistema de archivos, escritura atomica."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pred_engine.comun.logger import get_logger
from pred_engine.optimizacion.control_reanudacion.contratos import (
    SCHEMA_VERSION,
    ManifiestoCorrida,
)
from pred_engine.optimizacion.control_reanudacion.errores import (
    ManifiestoAusenteError,
    ManifiestoCorruptoError,
    VersionManifiestoError,
)

_logger = get_logger(__name__)

_NOMBRE = "manifiesto.json"


def escribir_atomico(ruta: Path, contenido: str) -> None:
    """Escribe a `ruta.tmp` y reemplaza el destino solo si termino bien."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    temporal.write_text(contenido, encoding="utf-8")
    os.replace(temporal, ruta)


class AlmacenManifiestosFs:
    """Implementa `AlmacenManifiestos` sobre un directorio raiz."""

    def __init__(self, raiz: str | Path) -> None:
        self._raiz = Path(raiz)

    def ruta_de(self, run_id: str) -> Path:
        return self._raiz / run_id / _NOMBRE

    def existe(self, run_id: str) -> bool:
        return self.ruta_de(run_id).is_file()

    def guardar(self, manifiesto: ManifiestoCorrida) -> None:
        ruta = self.ruta_de(manifiesto.run_id)
        payload = manifiesto.model_dump_json(indent=2)
        escribir_atomico(ruta, payload + "\n")
        _logger.info(
            "Manifiesto guardado run_id=%s estado=%s trials=%d",
            manifiesto.run_id,
            manifiesto.estado.value,
            manifiesto.n_trials_finalizados,
        )

    def cargar(self, run_id: str) -> ManifiestoCorrida:
        ruta = self.ruta_de(run_id)
        if not ruta.is_file():
            _logger.error("Manifiesto ausente run_id=%s ruta=%s", run_id, ruta)
            raise ManifiestoAusenteError(f"no existe manifiesto para {run_id}")
        try:
            bruto = json.loads(ruta.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _logger.error("Manifiesto corrupto run_id=%s: %s", run_id, exc)
            raise ManifiestoCorruptoError(f"manifiesto corrupto para {run_id}") from exc
        if not isinstance(bruto, dict):
            raise ManifiestoCorruptoError(f"manifiesto no es un objeto JSON: {run_id}")
        version = bruto.get("schema_version")
        if version != SCHEMA_VERSION:
            _logger.error(
                "Version de manifiesto no soportada run_id=%s version=%s",
                run_id,
                version,
            )
            raise VersionManifiestoError(
                f"schema_version={version!r} no soportada (esperada {SCHEMA_VERSION})"
            )
        try:
            manifiesto = ManifiestoCorrida.model_validate(bruto)
        except Exception as exc:
            raise ManifiestoCorruptoError(f"manifiesto invalido para {run_id}") from exc
        _logger.info(
            "Manifiesto leido run_id=%s estado=%s",
            manifiesto.run_id,
            manifiesto.estado.value,
        )
        return manifiesto

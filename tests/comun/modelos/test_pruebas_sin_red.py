"""La sesion de pruebas carga Chronos-2 solo desde la cache, sin red.

Con el extra `foundation` instalado, las pruebas del modelo real llaman al
cargador directamente; `tests/conftest.py` fija `HF_HUB_OFFLINE=1` para que
nunca descarguen pesos.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from pred_engine.comun.modelos.modelos_fundacionales import CHRONOS2_ZERO_SHOT


def test_la_sesion_no_puede_descargar_pesos(tmp_path: Path) -> None:
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    if importlib.util.find_spec("huggingface_hub") is None:
        return  # sin el extra no hay cliente de Hugging Face que descargue
    from huggingface_hub import constants, hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    assert constants.HF_HUB_OFFLINE
    # Una cache vacia sin red falla; con red, la descarga terminaria sin error.
    with pytest.raises(LocalEntryNotFoundError):
        hf_hub_download(
            CHRONOS2_ZERO_SHOT.model_id,
            "config.json",
            revision=CHRONOS2_ZERO_SHOT.revision,
            cache_dir=tmp_path,
        )
    assert not any(tmp_path.rglob("config.json"))

"""El artefacto consultivo conserva el marco intacto."""

from __future__ import annotations

import pandas as pd

from pred_engine.comun.modelos import DiagnosticEntry, HeaderDiagnostic
from pred_engine.ingesta.sonda import DiagnosticArtifact


def test_artefacto_congela_marco_y_diagnostico() -> None:
    marco = pd.DataFrame({"sku_id": ["A"], "timestamp": ["2024-01-01"]})
    reporte = HeaderDiagnostic(
        status="accepted",
        diagnostic=(
            DiagnosticEntry(
                field="schema",
                severity="info",
                message="ok",
            ),
        ),
    )
    artefacto = DiagnosticArtifact(frame=marco, diagnostic=reporte)
    assert artefacto.frame is marco
    assert artefacto.diagnostic.status == "accepted"


def test_no_hay_rename_ni_drop_en_el_modulo() -> None:
    import pred_engine.ingesta.sonda as sonda_pkg

    assert "apply_header_mapping" not in sonda_pkg.__all__

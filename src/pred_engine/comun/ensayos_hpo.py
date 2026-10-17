"""Lectura de los ensayos de un estudio HPO persistido, sin Optuna.

Cada estudio guarda sus ensayos en ``<hpo>/<estudio>/backend.jsonl`` con el
formato del backend (una linea por ensayo, ver ``adaptador_optuna``). Quien solo
quiere *leer* esa evidencia -- la plataforma, la tesis, un cuaderno -- no debe
necesitar Optuna ni conocer su vocabulario: este modulo la devuelve como
``Trial``, el mismo contrato que usan las estrategias, y no importa ``optuna``
(una prueba lo comprueba en un proceso aparte).

Lo que cierra la evidencia de cada ensayo:

* ``podado`` y ``fallido`` se distinguen por su estado y ambos llevan el
  ``motivo`` (p. ej. ``poda_semantica:prediccion_nula`` o
  ``ventana=1 ajuste_fallido: AjusteModeloError: ...``).
* Los estudios de la familia fundacional no existen: no optimiza, no hay ensayos.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pred_engine.comun.dataclasses.hpo import EstadoTrial, Trial

ARCHIVO_ENSAYOS = "backend.jsonl"
ARCHIVO_MANIFIESTO = "manifiesto.json"

# Estados del backend (Optuna) -> vocabulario del motor. WAITING y RUNNING no se
# persisten hoy (un ensayo en curso se vuelve a ejecutar al reanudar), pero un
# archivo de otra version podria traerlos.
_ESTADO_DESDE_BACKEND: Mapping[str, EstadoTrial] = {
    "WAITING": "pendiente",
    "RUNNING": "corriendo",
    "COMPLETE": "completado",
    "PRUNED": "podado",
    "FAIL": "fallido",
}


class EnsayosIlegiblesError(ValueError):
    """El archivo de ensayos existe pero no se puede interpretar."""


def _leer_manifiesto(carpeta: Path) -> Mapping[str, Any]:
    ruta = carpeta / ARCHIVO_MANIFIESTO
    if not ruta.is_file():
        return {}
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EnsayosIlegiblesError(f"{ruta}: manifiesto ilegible: {exc}") from exc
    return datos if isinstance(datos, dict) else {}


def _finito(valor: Any) -> float | None:
    if isinstance(valor, bool) or not isinstance(valor, int | float):
        return None
    return float(valor) if math.isfinite(valor) else None


def _ensayo(
    linea: str, *, ruta: Path, numero: int, carpeta: Path, manifiesto: Mapping[str, Any]
) -> Trial:
    try:
        datos = json.loads(linea)
        estado = _ESTADO_DESDE_BACKEND[datos["estado_optuna"]]
        atributos = datos.get("user_attrs") or {}
        configuracion = dict(datos.get("params") or {})
    except (ValueError, KeyError, TypeError) as exc:
        raise EnsayosIlegiblesError(
            f"{ruta}:{numero + 1}: ensayo ilegible ({type(exc).__name__}: {exc})"
        ) from exc
    motivo = atributos.get("motivo")
    return Trial(
        id=str(atributos.get("trial_id") or f"{carpeta.name}-{numero:04d}"),
        configuracion=configuracion,
        estado=estado,
        valor=_finito(datos.get("valor")),
        n_ventanas=int(atributos.get("n_ventanas", 0)),
        motivo=None if motivo is None else str(motivo),
        metrica_objetivo=str(manifiesto.get("metrica_objetivo", "")),
        familia=str(atributos.get("familia") or manifiesto.get("familia", "")),
        sku_id=atributos.get("sku_id") or manifiesto.get("sku_id"),
        timestamp=atributos.get("timestamp"),
    )


def leer_ensayos(carpeta_estudio: str | Path) -> tuple[Trial, ...]:
    """Ensayos del estudio, en el orden en que se cerraron.

    Una carpeta sin ``backend.jsonl`` (estudio que no llego a cerrar ningun
    ensayo, o que no existe) no tiene ensayos: devuelve una tupla vacia.

    Raises:
        EnsayosIlegiblesError: una linea no es JSON, no trae ``estado_optuna`` o
            su estado no se conoce. No se omite en silencio: seria perder evidencia.
    """
    carpeta = Path(carpeta_estudio)
    ruta = carpeta / ARCHIVO_ENSAYOS
    if not ruta.is_file():
        return ()
    manifiesto = _leer_manifiesto(carpeta)
    try:
        lineas = [
            linea for linea in ruta.read_text(encoding="utf-8").splitlines() if linea
        ]
    except OSError as exc:
        raise EnsayosIlegiblesError(f"{ruta}: {exc}") from exc
    return tuple(
        _ensayo(linea, ruta=ruta, numero=i, carpeta=carpeta, manifiesto=manifiesto)
        for i, linea in enumerate(lineas)
    )


def buscar_estudio(raiz_hpo: str | Path, familia: str, sku_id: str) -> Path | None:
    """Carpeta del estudio de ``familia`` y ``sku_id`` bajo ``raiz_hpo``.

    Los estudios clasicos y de ML se llaman ``<familia>-<sku>-<sesion>``, pero los
    de DL ``dl-<huella>``: el nombre no basta, asi que se busca por el manifiesto.
    Util cuando la unidad fallo y no hay ``estudio_hpo`` en su resultado. Si hay
    varios, el primero por nombre (la carpeta de una corrida es de una sola
    configuracion).
    """
    raiz = Path(raiz_hpo)
    if not raiz.is_dir():
        return None
    for carpeta in sorted(p for p in raiz.iterdir() if p.is_dir()):
        manifiesto = _leer_manifiesto(carpeta)
        if manifiesto.get("familia") == familia and manifiesto.get("sku_id") == sku_id:
            return carpeta
    return None


def ensayo_a_dict(ensayo: Trial) -> dict[str, Any]:
    """El ensayo con la forma de ``TrialRow`` del contrato ``sku_selection``."""
    return {
        "id": ensayo.id,
        "configuracion": dict(ensayo.configuracion),
        "estado": ensayo.estado,
        "valor": ensayo.valor,
        "n_ventanas": ensayo.n_ventanas,
        "motivo": ensayo.motivo,
        "timestamp": ensayo.timestamp,
    }

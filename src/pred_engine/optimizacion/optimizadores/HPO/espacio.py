"""DSL del espacio de busqueda: la pieza que hace el HPO agnostico de familia.

Lo que diferencia a clasicos/ML/DL no es el algoritmo de busqueda, es LA
FORMA del espacio: enteros pequenos acotados (clasicos), mixto con escala
log (ML), mixto + condicionales jerarquicos (DL). Este modulo soporta las
tres formas con el mismo vocabulario.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from pred_engine.optimizacion.optimizadores.HPO.errores import EspacioInvalidoError


@dataclass(frozen=True, slots=True)
class Entero:
    nombre: str
    bajo: int
    alto: int
    paso: int = 1

    def __post_init__(self) -> None:
        if self.bajo > self.alto:
            raise EspacioInvalidoError(f"{self.nombre}: bajo > alto")
        if self.paso < 1:
            raise EspacioInvalidoError(f"{self.nombre}: paso debe ser >= 1")

    def muestrear(self, rng: np.random.Generator) -> int:
        n_pasos = (self.alto - self.bajo) // self.paso + 1
        offset = int(rng.integers(0, n_pasos))
        return self.bajo + offset * self.paso


@dataclass(frozen=True, slots=True)
class Flotante:
    nombre: str
    bajo: float
    alto: float
    log: bool = False

    def __post_init__(self) -> None:
        if self.bajo > self.alto:
            raise EspacioInvalidoError(f"{self.nombre}: bajo > alto")
        if self.log and self.bajo <= 0:
            raise EspacioInvalidoError(f"{self.nombre}: log=True exige bajo > 0")

    def muestrear(self, rng: np.random.Generator) -> float:
        if self.log:
            log_bajo, log_alto = np.log(self.bajo), np.log(self.alto)
            return float(np.exp(rng.uniform(log_bajo, log_alto)))
        return float(rng.uniform(self.bajo, self.alto))


@dataclass(frozen=True, slots=True)
class Categorico:
    nombre: str
    opciones: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.opciones:
            raise EspacioInvalidoError(f"{self.nombre}: opciones no puede estar vacio")

    def muestrear(self, rng: np.random.Generator) -> Any:
        indice = int(rng.integers(0, len(self.opciones)))
        return self.opciones[indice]


@dataclass(frozen=True, slots=True)
class Ordinal:
    """Categorico con orden explicito, codificado como el indice entero de
    `niveles` (0..len(niveles)-1).

    TPE trata un entero como continuo ordenado -- justo la propiedad que un
    ordinal necesita (ej. 'bajo' < 'medio' < 'alto') y que un `Categorico`
    (universo intercambiable, sin nocion de cercania) no preserva. El nivel
    semantico se recupera indexando `niveles` con el entero muestreado; ver
    `nivel_de`.
    """

    nombre: str
    niveles: tuple[Any, ...]

    def __post_init__(self) -> None:
        if len(self.niveles) < 2:
            raise EspacioInvalidoError(f"{self.nombre}: ordinal requiere >= 2 niveles")

    def muestrear(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, len(self.niveles)))


def nivel_de(parametro: Ordinal, indice: int) -> Any:
    """Decodifica el indice entero muestreado de vuelta al nivel semantico."""
    return parametro.niveles[indice]


Parametro = Entero | Flotante | Categorico | Ordinal


@dataclass(frozen=True, slots=True)
class Condicion:
    parametro: str
    padre: str
    valores: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class EspacioBusqueda:
    parametros: tuple[Parametro, ...]
    condiciones: tuple[Condicion, ...] = ()
    restricciones: tuple[Callable[[Mapping[str, Any]], bool], ...] = ()

    def __post_init__(self) -> None:
        nombres = [p.nombre for p in self.parametros]
        if len(nombres) != len(set(nombres)):
            raise EspacioInvalidoError("nombres de parametro duplicados en el espacio")
        indice_de = {nombre: i for i, nombre in enumerate(nombres)}
        for condicion in self.condiciones:
            if condicion.parametro not in indice_de:
                raise EspacioInvalidoError(
                    f"condicion referencia parametro inexistente: {condicion.parametro}"
                )
            if condicion.padre not in indice_de:
                raise EspacioInvalidoError(
                    f"condicion referencia padre inexistente: {condicion.padre}"
                )
            if condicion.padre == condicion.parametro:
                raise EspacioInvalidoError("un parametro no puede depender de si mismo")
            if indice_de[condicion.padre] > indice_de[condicion.parametro]:
                raise EspacioInvalidoError(
                    f"'{condicion.padre}' debe declararse antes que "
                    f"'{condicion.parametro}' en `parametros` "
                    "(el muestreo es secuencial)"
                )

    def _condiciones_por_parametro(self) -> dict[str, Condicion]:
        return {c.parametro: c for c in self.condiciones}

    def nombres_activos(self, configuracion: Mapping[str, Any]) -> tuple[str, ...]:
        condiciones = self._condiciones_por_parametro()
        activos = []
        for parametro in self.parametros:
            condicion = condiciones.get(parametro.nombre)
            if condicion is None:
                activos.append(parametro.nombre)
                continue
            if configuracion.get(condicion.padre) in condicion.valores:
                activos.append(parametro.nombre)
        return tuple(activos)

    def muestrear(
        self, rng: np.random.Generator, *, intentos_max: int = 200
    ) -> dict[str, Any]:
        for _ in range(intentos_max):
            configuracion: dict[str, Any] = {}
            condiciones = self._condiciones_por_parametro()
            for parametro in self.parametros:
                condicion = condiciones.get(parametro.nombre)
                if condicion is not None:
                    if configuracion.get(condicion.padre) not in condicion.valores:
                        continue
                configuracion[parametro.nombre] = parametro.muestrear(rng)
            if self.es_valida(configuracion):
                return configuracion
        raise EspacioInvalidoError(
            f"no se pudo muestrear una configuracion valida en {intentos_max} "
            "intentos (las restricciones descartan casi todo el espacio)"
        )

    def es_valida(self, configuracion: Mapping[str, Any]) -> bool:
        return all(restriccion(configuracion) for restriccion in self.restricciones)

    def descripcion_canonica(self) -> dict[str, Any]:
        return descripcion_canonica(self)


def _etiqueta_restriccion(restriccion: Callable[..., Any]) -> str:
    # Las restricciones vigentes son callables (a menudo lambdas). Un
    # nombre calificado basta para funciones definidas; el bytecode
    # distingue lambdas distintas sin exigir un DSL extra de predicados.
    nombre = getattr(restriccion, "__qualname__", "") or getattr(
        restriccion, "__name__", ""
    )
    modulo = getattr(restriccion, "__module__", "") or ""
    if nombre and "<lambda>" not in nombre:
        return f"{modulo}.{nombre}" if modulo else nombre
    codigo = getattr(restriccion, "__code__", None)
    if codigo is None:
        digest = hashlib.sha256(repr(restriccion).encode("utf-8")).hexdigest()[:16]
        return f"anon:{digest}"
    digest = hashlib.sha256(codigo.co_code).hexdigest()[:16]
    return f"lambda:{digest}"


def _parametro_canonico(parametro: Parametro) -> dict[str, Any]:
    if isinstance(parametro, Entero):
        return {
            "tipo": "entero",
            "nombre": parametro.nombre,
            "bajo": parametro.bajo,
            "alto": parametro.alto,
            "paso": parametro.paso,
        }
    if isinstance(parametro, Flotante):
        return {
            "tipo": "flotante",
            "nombre": parametro.nombre,
            "bajo": parametro.bajo,
            "alto": parametro.alto,
            "log": parametro.log,
        }
    if isinstance(parametro, Categorico):
        return {
            "tipo": "categorico",
            "nombre": parametro.nombre,
            "opciones": list(parametro.opciones),
        }
    if isinstance(parametro, Ordinal):
        return {
            "tipo": "ordinal",
            "nombre": parametro.nombre,
            "niveles": list(parametro.niveles),
        }
    raise TypeError(f"tipo de parametro no soportado: {type(parametro)!r}")


def descripcion_canonica(espacio: EspacioBusqueda) -> dict[str, Any]:
    """Representacion JSON-serializable del espacio para la huella 2.9."""
    return {
        "parametros": [_parametro_canonico(p) for p in espacio.parametros],
        "condiciones": [
            {
                "parametro": c.parametro,
                "padre": c.padre,
                "valores": list(c.valores),
            }
            for c in espacio.condiciones
        ],
        "restricciones": [_etiqueta_restriccion(r) for r in espacio.restricciones],
    }

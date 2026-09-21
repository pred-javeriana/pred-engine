# ADR-012: TPE Sampler delegado a Optuna, no reimplementado

**Date:** 2026-09-21
**Status:** Accepted
**Notion ADR:** ADR-02-006
**Notion Task:** TASK-HPO-4.0-A1 / A2

## Context

El Modulo 2.4 (HPO) de Notion especifica un Tree-structured Parzen
Estimator (TPE): separar evaluaciones pasadas en "buenas"/"malas" segun un
umbral, estimar densidades `l(θ)`/`g(θ)` (KDE o histogramas) para cada
grupo, y proponer la configuracion que maximiza `l(θ)/g(θ)`, con soporte
para espacios mixtos (numericos, categoricos, ordinales, jerarquicos) y
warm start para SKUs con pocas ventanas informativas ("lumpy").

Para cuando este ADR se escribe, `muestreadores.py::construir_muestreador_tpe`
ya envuelve `optuna.samplers.TPESampler` en vez de implementar la
separacion/estimacion de densidades desde cero, y `optuna>=4.0` ya es
dependencia runtime del proyecto (usada tambien por ASHA via
`adaptador_optuna.py`, ver ADR-013). Este ADR formaliza esa decision
retroactivamente.

## Decision

Se usa `optuna.samplers.TPESampler(seed, multivariate=True, n_startup_trials,
n_ei_candidates)` como implementacion del algoritmo TPE. El motor de HPO
(`estudio.py`, `asha.py`, `contratos.py`) no conoce Optuna directamente: solo
`adaptador_optuna.py` importa la libreria (ver docstring del propio archivo).
`TPESampler` ya implementa la separacion buena/mala, la estimacion de
densidades (KDE) y el manejo de espacios mixtos/condicionales que la
especificacion de Notion pide construir.

Los espacios mixtos se codifican en `espacio.py`:

- Numericos → `Entero`/`Flotante` (`Flotante(log=True)` para escala
  logaritmica).
- Categoricos sin orden → `Categorico`.
- **Ordinales** (con orden, ej. `'bajo' < 'medio' < 'alto'`) → `Ordinal`,
  codificado como el **indice entero** de sus niveles (`0..len(niveles)-1`),
  no como `Categorico`: un entero preserva la nocion de cercania entre
  niveles adyacentes que TPE aprovecha al tratarlo como continuo ordenado,
  mientras que `Categorico` es un universo intercambiable sin esa nocion.
  El nivel semantico se recupera con `nivel_de(parametro, indice)`.
- Jerarquicos/condicionales → `Condicion` (`padre`, `valores`, `parametro`
  dependiente), ya soportado antes de este ADR.

Warm start (`adaptador_optuna.py::semillas_desde_historico` +
`inyectar_historico`, expuesto como `EstudioHPO.agregar_trials_historicos`
en el contrato) inyecta trials de una corrida EXTERNA (otro segmento de la
misma serie, util para SKUs lumpy) via el mismo mecanismo que restaura un
checkpoint (`study.add_trial`), sin reevaluarlos.

## Rationale

- `TPESampler` es la implementacion de referencia de TPE (Bergstra et al.)
  y ya cubre separacion buena/mala, KDE, espacios mixtos/condicionales y
  multivariante -- reimplementarlo duplicaria un algoritmo ya maduro y
  probado, sin beneficio funcional.
- Mantiene el criterio de "90% de reutilizacion entre verticales": clasicos,
  ML y DL comparten el mismo sampler cambiando solo `EspacioBusqueda`.
- La frontera `contratos.py` (`Protocol`, sin `import optuna`) sigue vigente:
  si el dia de manana se cambia de backend, solo se reescribe
  `adaptador_optuna.py`.

## Consequences

- Los criterios de aceptacion de Notion redactados en terminos de "estimar
  `l(θ)`/`g(θ)` con KDE/histogramas propios" se satisfacen por composicion
  (Optuna los implementa), no por codigo propio en este repo.
- Limitacion heredada de Optuna, ya documentada en ADR-011 y reforzada aqui:
  al reanudar una corrida con `TPESampler` (no con `RandomSampler`), el RNG
  de su fase de arranque en frio no se resincroniza -- `adaptador_optuna.py`
  ahora emite un `_logger.warning` explicito en ese caso (antes era
  silencioso) para que la perdida de presupuesto de trials sea visible, en
  vez de intentar un replay del RNG cuya correccion no se puede garantizar
  (el conteo de muestras de TPE por `ask()` no es necesariamente constante).
- `Ordinal` es aditivo: no cambia el comportamiento de espacios existentes
  que no lo usan (`Entero`/`Flotante`/`Categorico`/`Condicion` intactos).

## Alternatives Considered

- **Reimplementar TPE con `scipy.stats.gaussian_kde`:** rechazado. `scipy`
  ya llega transitivamente via Optuna, asi que no ahorra una dependencia, y
  duplicaria ~cientos de lineas de logica ya cubierta y probada por Optuna,
  ademas de perder actualizaciones/fixes upstream.
- **GP-BO (Bayesian Optimization con procesos gaussianos):** rechazado por
  el propio texto de Notion (2.4 HPO): rinde mal con variables no numericas
  o jerarquias condicionales, justo el caso de PRED (clasicos/ML/DL con
  espacios mixtos).
- **Reescribir el RNG-replay de `_sincronizar_sampler_aleatorio` para
  cubrir tambien `TPESampler`:** rechazado por ahora; el mecanismo que
  funciona para `RandomSampler` (avanzar el RNG con un estudio auxiliar)
  no tiene garantia de correccion equivalente para `TPESampler`, cuyo
  consumo de RNG por `ask()` puede depender del historial de trials. Se
  prefiere una advertencia honesta a una "solucion" no verificable.

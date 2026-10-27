# Verificación local del motor (`make motor-local`)

CI no corre el motor sobre datos reales: una corrida real tarda demasiado para
cada PR. Antes de pedir revisión, quien abre el PR corre `make motor-local` en
su equipo y publica el recibo como comentario del PR. El control `motor-local`
de CI lee ese comentario y no corre el motor. El trabajo `run` de CI sigue
corriendo el nivel `humo` en cada PR.

```
rutas cambiadas ──> niveles ──> corridas ──> recibo ──> comentario del PR ──> estado motor-local
(git diff)          (mapa)      (--workers 2)  (local_data/recibos)              (CI, segundos)
```

## Uso

Requisitos: `uv sync --frozen --extra dev`, una copia de `pred-platform` en
`../pred-platform` (crea la base central y tiene el caso de integración) y,
para publicar, `gh` con sesión iniciada.

1. Haga commit de todo: el comando se niega a correr con cambios sin commit,
   porque el recibo nombra un solo commit.
2. Vea qué niveles exige la rama: `make motor-local ARGS=--plan`.
3. Corra y publique: `make motor-local ARGS=--publicar`. Sin `gh`, corra
   `make motor-local` y pegue en el PR el recibo que imprime.

Opciones: `--base` (por defecto `origin/main`), `--plataforma RUTA`,
`--muestra22 RUTA` (por defecto `local_data/muestras/timing22.csv`) y
`--nivel NIVEL`, que agrega un nivel y nunca quita los que exigen las rutas.
Cada corrida usa una base nueva en `local_data/motor-local/<nivel>/` con 4
hebras de BLAS y `--workers 2`, así que cabe en un equipo de 4 núcleos y 8 GB.
Solo puede haber una corrida activa a la vez por máquina. El comando adquiere un
cerrojo en la ruta fija `/tmp/pred-motor-local.lock` (independiente de variables
como `TMPDIR`) antes de iniciar los niveles y lo libera al terminar. Si un proceso
previo fue terminado abruptamente (por ejemplo con `kill -9`), elimine el
cerrojo huérfano antes de volver a correr: `rmdir /tmp/pred-motor-local.lock`
(o `rm -rf /tmp/pred-motor-local.lock`).

## Niveles

`.github/motor-local.toml` asigna niveles a cada ruta. Gana la primera regla
que coincide, y el PR exige la unión de los niveles de sus rutas. Una ruta sin
regla exige todos los niveles, así una carpeta nueva no evita la corrida.

| Nivel | Rutas que lo exigen | Muestra | Pasa si | Tiempo medido (7900X, 4 núcleos) |
| --- | --- | --- | --- | --- |
| ninguno | `docs/**`, `**/*.md`, `tests/**`, `.github/**` y archivos de repositorio | ninguna | siempre | 0 |
| `humo` | M0 (`aumentacion/`), telemetría, `verify.py`, `comun/llm/`, `Makefile` | la corrida de CI: `tests/fixtures/semilla_inventario.csv` con M0 `mbb-directo` (8 SKU) | salida 0 u 8, telemetría y `verificar_corrida.py` | 30 s |
| `muestra22` | el resto de `src/pred_engine/` (L1-L4), `pyproject.toml`, `uv.lock`, `.python-version` | `timing22.csv`: 22 productos reales, `--datos-reales` y ajustes por defecto | salida 0 u 8, `verificar_corrida.py --datos-reales` y un veredicto por producto | 19 min |
| `plataforma` | `forecasting/persistencia_evidencia/` (junto con `muestra22`) | `tests/test_integracion_motor.py` de pred-platform con este motor | pytest termina sin fallas ni pruebas omitidas | 16 s |

La corrida de los 3.703 productos sigue siendo manual.

## Recibo

El recibo queda en `local_data/recibos/<commit>.txt`. Un recibo con
`resultado: fallo` también se publica, porque una corrida fallida informa.

```text
recibo-motor-local/1
repo: pred-engine
commit: 0915ede529be21176c43bf2bc551733caf18e887
niveles: muestra22
resultado: ok
motor: 0915ede529be21176c43bf2bc551733caf18e887
equipo: 4 nucleos, 8 GiB, --workers 2, hebras 4
fecha: 2026-10-27T18:45Z

muestra22: ok salida=8 duracion_s=1128 muestra=m22-941198f5af6e veredictos=EXPLORATORIO:3,FALLO_TECNICO:1,VALIDADO:18 huella=f26edda9c213
```

`muestra` identifica la entrada: los primeros 12 dígitos del SHA-256 del
archivo (para `plataforma`, del blob git del caso). `huella` son los primeros
12 dígitos del SHA-256 de las líneas `sku:veredicto:candidato_campeon`
ordenadas; quien revisa puede correr el mismo nivel y comparar `huella` y
`veredictos`. `motor` es el commit del motor que corrió.

## Control `motor-local` de CI

`.github/workflows/recibo-motor.yml` corre al abrir o actualizar el PR y al
crear o editar un comentario. Usa el script y el mapa de la rama base, lee el
PR por el API y fija el estado `motor-local` del commit cabeza:

- `success` si las rutas del PR no exigen ningún nivel (`sin corrida: solo docs`).
- `success` si el recibo más reciente de un autor con permiso de escritura
  nombra la cabeza, tiene `resultado: ok` y cubre los niveles exigidos.
- `success` con un recibo de un commit anterior si todos los cambios desde ese
  commit son solo docs (`cambios posteriores solo docs`).
- `failure` en otro caso, por ejemplo `falta recibo muestra22 para 1a2b3c4`.

Un push con código nuevo deja el recibo vencido. Un rebase o un merge de
`main` también exigen un recibo nuevo. Para ver el estado sin fijarlo:
`python3 .github/scripts/recibo_motor.py --repo pred-javeriana/pred-engine --pr N`
con `GITHUB_TOKEN` en el entorno.

## Muestra `timing22.csv`

La muestra son datos reales de ventas y no se versiona. Pida `timing22.csv` al
equipo y cópielo en `local_data/muestras/timing22.csv`, o reconstrúyalo desde
el panel real preparado `panel_listo.csv` con
`uv run python .github/scripts/muestra22.py panel_listo.csv local_data/muestras/timing22.csv`.
El script quita los productos sin demanda positiva hasta t* y toma 22 al azar
con semilla 17; debe imprimir `m22-941198f5af6e`. Si imprime otro id, el
recibo lo registra y quien revisa ve que la muestra es otra.

## Límites

El recibo prueba que una cuenta con escritura publicó que corrió un nivel sobre
un commit. No prueba que la corrida ocurrió ni quién la hizo, y esa cuenta
también puede fijar el estado por el API. Es un control de proceso: saltarse la
corrida queda como un acto deliberado, con nombre, en el historial del PR.

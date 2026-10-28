# Reanudación durable de L2/L3

RF-MOD-11 exige continuar unidades faltantes de una ejecución interrumpida sin reprocesar las
exitosas. Antes se recuperaban estudios HPO y unidades M3, pero L2/L3 se despachaban completas
en cada intento. Esto causaba los problemas reportados en platform
[#37](https://github.com/pred-javeriana/pred-platform/issues/37) y
[#38](https://github.com/pred-javeriana/pred-platform/issues/38).

## Garantía y almacenamiento

La CLI activa `UnitCheckpoints` en `runs/{run_id}/checkpoints/{L2,L3}/`. Una unidad exitosa guarda
un JSON tipado con su resultado y registro temporal, clave de contexto y suma SHA-256.
La escritura es atómica y sincroniza el archivo antes del evento `unit_finished`; en POSIX
también sincroniza el directorio. En Windows se abre el archivo con acceso de lectura/escritura,
porque sincronizar un descriptor abierto solo para lectura produce `Bad file descriptor`.
El callback se ejecuta dentro del proceso que realizó la unidad, por lo que también se preserva
una unidad que termina mientras el padre espera el resultado de otra.

L2 conserva selección, historia admisible y pronóstico de la reserva; L3 conserva la evaluación
Walk-Forward, sus ventanas, métricas y pronósticos originales. No guarda objetos de modelos ni
usa pickle para el checkpoint. El codec solo reconstruye tipos permitidos explícitamente.
Fallos y resultados incompletos no crean checkpoints reutilizables.

Al reanudar, el pipeline reconstruye los resultados exitosos y despacha únicamente las unidades
faltantes. Los registros restaurados conservan PID y tiempos originales y tienen `reused: true`
en `unidades.jsonl` y en su evento de finalización; no emiten `unit_started`.
Las exportaciones derivadas se reconstruyen incluyendo toda la evidencia. Los agregados de
paralelismo y el gráfico de telemetría excluyen esos tiempos históricos del trabajo del intento.

La clave incluye identidad del experimento, código, configuración, versiones y entrada efectiva
de la unidad. La CLI mantiene su protección anterior de `identidad.json` sobre la fuente y los
ajustes. La API de biblioteca es optativa: quien proporcione `UnitCheckpoints(identity=...)` debe
identificar también código, dependencias y configuración externa de sus estrategias y fábricas.
Una clave incompatible no se reutiliza y un archivo corrupto falla explícitamente sin sobrescribirlo.

## Recuperación de M3 sin archivos locales

Si se pierden `candidatos.json` y los checkpoints de L2, M3 busca primero el protocolo
congelado por la ejecución central ya validada. Comprueba entrada, código, ajustes y
partición antes de restaurar el manifiesto. No usa los candidatos reconstruidos para
buscar una publicación anterior ni cambia su identidad.

Una corrida `PUBLICADA` conserva toda su evidencia aunque un ajuste final falle durante
el reintento. Una corrida activa continúa con la receta congelada si faltan candidatos
por fallos técnicos del intento original o del reintento. Los candidatos reconstruidos
con éxito deben coincidir con esa receta. Una deriva de candidatos, protocolo o partición rechaza la corrida activa
con estado `RECHAZADA` y una causa durable, sin crear otro protocolo.

Si una ejecución ya tiene varios protocolos M3, el reintento falla explícitamente con
`la ejecucion tiene varios protocolos M3 durables`. No elige uno ni modifica su evidencia.
La recuperación no elimina publicaciones duplicadas existentes.

## Compatibilidad y límites

- Conservar la base central y el directorio completo de la corrida, incluidos HPO y M3.
- Solo se omiten unidades completas confirmadas. La unidad que estaba en curso vuelve a empezar;
  no se añaden checkpoints dentro de un ajuste o ventana Walk-Forward.
- No hay reconstrucción retroactiva de checkpoints desde resúmenes ni estados de platform.
  El mismo `run_id` con otra versión de código sigue siendo rechazado por la protección existente.
- HPO y M3 conservan sus reglas de identidad, estados terminales y publicación congelada.
  Un reintento no vuelve compatible una entrada distinta ni garantiza resolver fallos deterministas.
- Platform necesita la corrección complementaria para no reproducir eventos anteriores como
  trabajo vivo y para esperar los artefactos finales después de la publicación M3.
- Sin cambios de métricas, lógica de pronóstico, presupuestos HPO, Walk-Forward, contratos o M4/D3.

## Pruebas

```bash
uv run --frozen pytest tests/test_unit_checkpoints.py tests/test_run_chart.py --no-cov
uv run --frozen pytest -m "not slow"
uv run --frozen pytest -m slow --no-cov
uv run --frozen python -m pred_engine.verify
```

Las regresiones cubren restauración con uno y dos procesos, interrupción en L2/L3, fallo aislado,
igualdad de pronósticos/métricas/evidencia, cambio de identidad o entrada, corrupción, error de
persistencia y durabilidad en un proceso hijo antes de que el padre consuma su resultado.
El cambio se verificó además con el worker real de platform y la CLI, dos SKU SARIMA y fallo
transitorio, cierre normal, detención del operador y caída abrupta: cada segundo intento reutilizó tres unidades y ejecutó
solo la evaluación faltante, finalizando sin duplicar tareas ni selecciones.

Verificación local del 26 de octubre de 2026 en Linux: 1185 pruebas de la suite principal aprobadas,
cobertura 94,01 %, 81 pruebas lentas aprobadas, lockfile, Ruff, formato y Pyright correctos.
`pred_engine.verify` pasó y el smoke M0–M3 publicó correctamente, con telemetría y artefactos/base
central verificados. La suite principal omite Chronos real sin su extra y el dataset oficial ausente.

También se ejecutaron nativamente en Windows NT 10.0.26300.0 y CPython 3.12.12 las 14 pruebas de
`test_unit_checkpoints.py` y `test_run_chart.py`, incluidas las de dos procesos. La CLI real y el
worker de platform se probaron con uno y dos procesos, fallo transitorio, cierre normal,
detención/reanudación y muerte abrupta del worker. La corrección complementaria de platform espera
al cierre del árbol completo de procesos antes de reanudar.
El job `windows-recovery` conserva las regresiones nativas en la CI del motor.

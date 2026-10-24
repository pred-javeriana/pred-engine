# ADR-021: Evaluación reservada, recuperación M3 y base única de plataforma

**Fecha:** 2026-10-18
**Estado:** Aceptada
**Referencias:** ADR-03-003 a ADR-03-011 del Hub; tareas 3.0 a 3.7.

## Contexto

El motor ya entrega candidatos completos desde M2. Faltaban el generador de
ventanas reservadas, la coordinación de M3, la recuperación durable y la
conexión con el pipeline. Los textos del Hub describen variantes distintas
para incorporar observaciones reservadas y para emitir evidencia estadística.

## Decisiones

1. **Ajuste único y causalidad por origen.** Cada candidato se ajusta una vez
   con historia hasta t*. En cada origen incorpora únicamente las observaciones
   disponibles hasta ese origen, sin reestimar parámetros ni ejecutar HPO.
   Modificar observaciones posteriores al origen no altera su pronóstico.
2. **Ventanas diarias completas.** H se fija antes de abrir la reserva. Los
   orígenes diarios van desde t* hasta el último origen con H días reservados
   completos. El paso del HPO no modifica ese calendario. Nunca se acorta H.
3. **Métricas descriptivas y evidencia estadística complementaria.** MAE,
   RMSE, MASE, ME, razón frente a Seasonal Naive y cobertura se conservan como
   evidencia principal. HLN-DM con corrección BH se mantiene activada por
   defecto, aparte del veredicto descriptivo. Solo usa datos reales y exige
   `n_min_dm` orígenes comunes, predeclarados en configuración. El valor
   inicial es 30. El horizonte de la prueba es H, no el paso del HPO.
   Con pocos orígenes se publica una causa, no un p-valor. Los datos sintéticos
   o aumentados nunca participan en la prueba.
4. **Una sola base.** Todos los resultados M3 van a la base SQLite de la
   plataforma. El motor recibe una conexión abierta. El CLI busca
   `--db-path`, `PRED_DB_PATH` o `{data-root}/pred.db`, en ese orden. No crea
   una base alternativa cuando la de plataforma falta.
5. **Migraciones compartidas.** El motor conserva el texto canónico de los
   esquemas M3. La plataforma incorpora copias exactas como migraciones 0005
   a 0009. La 0007 conserva la procedencia M1, la 0008 distingue el origen
   desconocido de una declaración real y la 0009 expone esa marca en la vista
   publicada, sin cambiar los textos ya publicados. Ambos usan `PRAGMA user_version`. El CLI aplica migraciones M3
   pendientes sobre una base de plataforma existente, sin alterar las
   migraciones 0001 a 0004 ni aceptar versiones desconocidas.
6. **Unidad durable.** Cada origen confirma pronóstico, estado, causa si
   falla, semilla, versión y emisión en una transacción. Una unidad confirmada
   o fallida no se recalcula. La recuperación retoma la fase pendiente.
7. **Protocolo congelado.** La identidad incorpora la receta completa y la
   configuración predeclarada. El manifiesto, el instante de apertura y la
   procedencia se conservan en SQLite. Reanudar no vuelve a emitir el handoff.
   El origen real o sintético declarado de una ingesta no cambia entre corridas.
   Una ausencia queda desconocida y no admite inferencia ni VALIDADO. La carga
   web sigue recibiendo solo el archivo, sin declarar que los datos sean reales.
8. **Publicación.** M3 publica métricas por candidato y ventana, selección,
   veredictos y resúmenes de cartera precalculados. SQLite es la fuente de
   verdad. Parquet y CSV son exportaciones derivadas. El directorio final
   aparece después de preparar el conjunto completo y M4 solo lee corridas
   PUBLICADA. M4 no evalúa modelos ni calcula métricas evaluativas.

## Contradicciones documentales

- ADR-03-001 describe reajustes expansivos. TASK-EVAL-3.3-A2 exige invariancia
  ante cambios en toda la reserva. La decisión 1 precisa la garantía vigente:
  invariancia respecto del futuro de cada origen, no de todo el periodo
  reservado. Aplica la recomendación de ADR-03-004.
- ADR-03-008 adoptó veredictos solo descriptivos. [ADR-03-011](https://app.notion.com/p/3fdfbd1bee468138b996f52cc4f31b3e)
  conserva esas reglas y añade HLN-DM/BH como evidencia complementaria con
  mínimo configurable, nunca como regla del veredicto.
- El aviso de 3.0 sobre módulos M2 vacíos está desactualizado. El control HPO
  existente se reutiliza, no se reemplaza por otro coordinador.
- El comentario histórico del esquema inicial M3 menciona una migración 0002.
  Ese texto se mantiene inmutable. En la plataforma actual la migración 0002
  pertenece al informe de ingesta; la evidencia M3 se instala desde 0005.

El Hub no se modifica como parte de esta implementación.

## Límites

La familia campeona se elige sobre la misma reserva que produce su puntaje.
La evidencia es retrospectiva, no una estimación no sesgada de desempeño
futuro. `VALIDADO` conserva su significado descriptivo. Las ventanas diarias
se solapan y HLN-DM usa su horizonte H para la corrección correspondiente.
La corrección BH se aplica entre todos los SKU efectivamente probados.

La aceptación personal del líder de ingeniería, la revisión de otro fundador y la fusión
son condiciones de cierre del proyecto, no estados que el motor pueda emitir.

# API: coordinación de evaluación M3

Punto de entrada: `pred_engine.forecasting.control_reanudacion`.

```python
ControladorEvaluacion(
    conn: sqlite3.Connection,
    directorio: str | Path,
    *,
    run_id_m2: str,
    codigo_version: str,
    datos_sinteticos: bool,
    data_root: str | Path | None = None,
    politica: PoliticaSeleccion = POLITICA_INICIAL,
    configuracion_m2: Mapping[str, Any] | None = None,
    fabricas: Mapping[str, FabricaAdaptador] = FABRICAS,
)
```

La conexión debe estar abierta, sin transacción activa y con las migraciones
M3 aplicadas. `directorio` es la carpeta de la corrida M2. El llamador conserva
la propiedad de la conexión y la cierra al terminar. `data_root` es la raíz
efectiva que permite resolver las rutas relativas registradas por M1; el CLI
la pasa explícitamente. La identidad CSV original no se sustituye por la huella
Parquet cuando la plataforma ya conoce esa ingesta.

La instancia implementa `RetrospectiveValidator`:

```python
resultado: ValidationArtifact = controlador(evaluation: EvaluationArtifact)
```

- Consume todos los candidatos de `evaluation.fitting`, incluso si una unidad
  de la reevaluación L3 falló. No trata el puntaje HPO como veredicto final.
- Produce o reutiliza `candidatos.json` antes de abrir la reserva. Su contenido
  y emisión quedan congelados; una receta diferente bajo el mismo identificador
  no se usa como reanudación. Si existe un protocolo durable, su manifiesto es
  la autoridad y se consulta antes de restaurar el archivo local.
- Deriva el contexto del Parquet M1 real y el corte común. Contrasta también
  `sku_class` con M1.
- Fija H, política completa, origen de datos, código, versiones y referencias
  M1/M2. Registra antes de abrir la reserva si la métrica HPO no es RMSE.
- El resultado contiene exactamente un veredicto para cada SKU del handoff.
  Los fallos de adaptación cuentan, aunque el SKU no tenga candidato válido.

`ValidationArtifact.audit_bundle` incluye `run_id_m3`, estado PUBLICADA,
origen de datos, H, número de unidades, fallos técnicos, exclusiones de datos
y rutas de exportación. Las ventanas sin contexto no se registran como fallos
del modelo; las ventanas sin objetivos completos no se puntúan ni se rellenan
con ceros. La ejecución de plataforma solo se declara terminada después de L4;
un error de esa etapa deja `detenida` y conserva el detalle L4.
Los errores de lote producen `LoteRechazadoError` y se registran RECHAZADA.
Una excepción de almacenamiento se propaga y conserva la última fase durable.

La biblioteca permite construir `Pipeline` sin este validador para ejecutar
etapas independientes. En ese caso L4 queda bloqueada por configuración,
no porque M3 carezca de implementación. El CLI siempre configura el validador
nativo sobre la base central existente. Un validador de biblioteca preexistente
conserva la cobertura de los SKU evaluados en L3; solo el nativo exige todos los
SKU del handoff. `identidad.json` fija la entrada y configuración antes de HPO,
independientemente de que un intento produzca o pierda `corrida.json`.

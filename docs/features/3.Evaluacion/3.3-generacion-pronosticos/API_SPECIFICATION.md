# API: generación de pronósticos reservados 3.3

Punto de entrada: `pred_engine.forecasting.generador_pronosticos`.

## Configuración y calendario

```python
ConfiguracionPronosticos(
    horizonte: int,
    codigo_version: str,
    min_historia: int = 7,
)
ventanas_reserva(corte: ReserveCut, horizonte: int) -> tuple[VentanaReserva, ...]
```

La configuración es inmutable. H debe ser un entero positivo, la versión no
puede estar vacía y `min_historia` debe ser un entero de al menos siete días.
`VentanaReserva` contiene `origen` y `fechas` diarias de longitud H.
`ventanas_reserva` rechaza cortes inconsistentes o con fracción distinta de 0,2.
Devuelve cero ventanas si H excede la reserva, nunca una ventana truncada.

## Generación

```python
generar_pronosticos(
    candidato: Candidato,
    serie: pd.Series,
    corte: ReserveCut,
    configuracion: ConfiguracionPronosticos,
    *,
    fabricas: Mapping[str, FabricaAdaptador] = FABRICAS,
    pipeline_fundacional: PipelineFundacional | None = None,
    confirmadas: Sequence[UnidadPronostico] = (),
    confirmar: Callable[[UnidadPronostico], None] | None = None,
) -> ResultadoPronosticos
```

- `candidato` procede del handoff validado por 3.2.
- `serie` lleva demanda con índice `DatetimeIndex` diario, sin zona horaria ni
  fechas repetidas. Se ordena sin modificar la entrada.
- El ajuste solo usa fechas <= t*. Cada pronóstico recibe las observaciones
  entre el primer día reservado y su origen. Las filas posteriores al final
  del corte no intervienen.
- `confirmadas` contiene unidades del mismo candidato, código y protocolo.
  Una unidad ya confirmada conserva sus valores y su fecha de emisión.
- `confirmar` recibe cada unidad nueva, incluidos los fallos, antes de seguir
  al próximo origen. Debe confirmar resultado y marcador en una transacción.
  Un error de almacenamiento interrumpe la ejecución y se propaga.

`ResultadoPronosticos` contiene `candidato` y `unidades`. Su propiedad `serie`
entrega un `SerieCandidato` aceptado por 3.4. Su propiedad `fallos` entrega
los fallos por origen.

## Unidad y fallo

`UnidadPronostico` contiene el candidato completo, `codigo_version`,
`emitido_en` en UTC, `pronostico: PronosticoFechado` y un `fallo` opcional.
Los valores nuevos del pronóstico son arreglos de solo lectura.

`FalloPronostico` contiene `sku`, `candidato_id`, `modelo`, `origen`, `campo`,
`codigo_error` y `mensaje`. Los códigos son:

- `ajuste_fallido`: no se pudo construir o ajustar el candidato.
- `pronostico_fallido`: el adaptador lanzó una excepción al pronosticar.
- `pronostico_invalido`: dimensión incorrecta o valores no finitos.
- `observaciones_ausentes`: no hay observaciones numéricas finitas hasta el origen.

Una unidad fallida lleva H valores NaN. No se imputan ceros. El error finito,
por grande que sea, no se convierte en fallo técnico.

## Exclusión

`separar_serie(sku, serie, corte, configuracion)` devuelve la historia como
arreglo y la reserva como serie fechada sin imputación. Puede lanzar
`SerieNoElegibleError`, con `sku` y `codigo`:

- `INSUFFICIENT_HISTORY`
- `CALENDAR_GAPS`
- `INVALID_HISTORY`
- `INSUFFICIENT_RESERVE`

La exclusión ocurre antes del ajuste. Una reserva ausente no se rellena con
ceros. Una unidad restaurada incompatible produce `ValueError` antes de
instanciar un modelo.

## Registro

Cada exclusión o fallo emite ERROR con SKU, candidato cuando corresponde,
código y causa. La generación terminada emite INFO con ventanas, restauradas
y fallidas. No se escriben series de demanda en los mensajes.

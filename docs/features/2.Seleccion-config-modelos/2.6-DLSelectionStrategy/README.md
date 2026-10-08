# Seleccion temporal de Deep Learning

`DLSelectionStrategy` adapta la familia `dl` al contrato comun de Module 2.
Recibe `SelectionRequest` y su perfil topologico, ordena las observaciones por
fecha y devuelve `SelectionResult`. La politica inicial sigue habilitando DL
**solo para `smooth` y `erratic`**. El registro es explicito:

```python
from pred_engine.optimizacion.optimizadores.modelos_deep_learning import (
    DLSelectionStrategy,
)
from pred_engine.optimizacion.router import StrategyRegistry

registro = StrategyRegistry()
registro.register("dl", DLSelectionStrategy(n_trials=12))
# Registrar tambien las otras familias que solicite la politica del router.
```

El router sigue devolviendo candidatos de varias familias, no un campeon.
No se agrega CLI, adaptador Parquet, catalogo fundacional ni evaluacion de Module 3.

## Predictor minimo

`comun.modelos.modelos_deep_learning.MLPForecaster` implementa el protocolo comun
`fit(y) -> self`, `predict(horizon) -> ndarray`. Es una red densa autoregresiva
con dos o mas capas ocultas tanh, salida lineal, dropout invertido, penalizacion
L2 y SGD por mini-batches. Usa NumPy, ya obligatorio, sin un nuevo framework,
descargas, GPU ni dependencias opcionales. No es un motor general de redes.

Cada ajuste reinicia pesos y normalizacion con una semilla local. La media y
escala proceden exclusivamente del prefijo de entrenamiento recibido. Cada
muestra contiene los `lags` valores anteriores a su objetivo. El entrenamiento
recorre mini-batches de ese prefijo en orden, sin dividirlo aleatoriamente. La
prediccion es recursiva, sin dropout, no negativa por defecto y no modifica el
estado ajustado. Un ajuste fallido invalida el modelo anterior.

`fabrica_dl(configuracion, seed=...)` permite reconstruir el predictor a partir
de los hiperparametros del ganador. Se debe ajustar despues con la historia
permitida: la seleccion no devuelve un modelo ya entrenado sobre toda la serie.

## Espacio y presupuesto

`EspacioDL` recibe rangos inclusivos `(minimo, maximo)`:

| Hiperparametro | Default | Papel |
| --- | --- | --- |
| `lags` | 3..14 | Ventana de entrada |
| `capas` | 2..3 | Capas ocultas |
| `unidades` | 8..32 | Ancho compartido por las capas ocultas |
| `epochs` | 10..50 | Epocas por ajuste |
| `batch_size` | 8..32 | Muestras por mini-batch |
| `learning_rate` | 0.001..0.05, log | Paso de SGD |
| `dropout` | 0..0.3 | Fraccion descartada en entrenamiento |
| `l2` | 1e-8..0.01, log | Regularizacion de pesos |

Todos los parametros estan activos: esta arquitectura no tiene bloques ni
embeddings condicionales. Se validan ambos extremos y se rechazan rangos
invertidos, no finitos o no entrenables. La restriccion combinada es
`numero_parametros * epochs <= costo_max` (100000 por defecto), donde se cuentan
pesos y sesgos. Acota trabajo por muestra; el costo total sigue dependiendo de
la longitud de la serie, el numero de ventanas y `n_trials`. La cota forma parte
de la huella de reanudacion. Un presupuesto que no permite ni el candidato
minimo falla antes de crear un estudio.

## Seleccion y evidencia

```python
import numpy as np
from pred_engine.optimizacion.optimizadores.modelos_deep_learning import (
    EspacioDL,
    seleccionar_configuracion_dl,
)

estudio = seleccionar_configuracion_dl(
    10 + np.sin(np.arange(40) * 2 * np.pi / 7),
    sku_id="SKU-1",
    espacio=EspacioDL(lags=(3, 4), unidades=(4, 8), epochs=(4, 8)),
    n_trials=3,
    min_train=10,
    horizonte=2,
    paso=5,
    metrica_objetivo="mae",
    seed=0,
)
assert estudio.mejor is not None
```

El selector devuelve directamente `ResultadoEstudio`, compartido con las otras
familias. No introduce otro protocolo de resultados ni usa AIC/BIC. Delega a
`seleccionar_con_hpo`: TPE propone, ASHA asigna/poda, Walk-Forward evalua prefijos
expansivos usando las mismas ventanas para todos los candidatos.

`DLSelectionStrategy` acepta los mismos parametros de espacio, presupuesto,
validacion, metrica, reglas, semilla y persistencia. Su `payload` contiene
`hiperparametros`, `metrica_objetivo`, `valor`, `n_ventanas`, `n_trials`,
`n_completados`, `n_podados`, `n_fallidos`, `seed` y `estudio_hpo`, la
referencia al estudio persistido (o `None` sin persistencia); los
estados y motivos por trial quedan en ese estudio. El payload solo lleva datos
serializables a JSON. El SKU, clase y perfil quedan en `SelectionResult`.
Una busqueda sin ganador completo genera `SelectionContractError`, no un
resultado vacio. La funcion de bajo nivel conserva el estudio sin ganador para
permitir inspeccionar los fallos.

RMSE es el default, como en los selectores clasico y ML: M3 elige la familia
campeona con la razon de RMSE frente a Seasonal Naive (ADR-03-007), asi que el
HPO optimiza el mismo error. `metrica_objetivo` permite las metricas comunes
MAE, RMSE, sMAPE y MASE (nombres en minuscula); `estacionalidad` configura el
escalado MASE, con default 1. Horizonte y paso tienen default 7. El minimo de
entrenamiento cubre el mayor lag mas cinco muestras; para MASE tambien debe
superar la estacionalidad. Se rechazan series no finitas o sin historia para
completar al menos cuatro ventanas. No se recorta el espacio automaticamente.

### Poda y fallos

`REGLAS_DL` es `ReglasPoda(habilitar_poda_semantica=False)`. Se conserva la poda
ASHA comun, con minimo de cuatro ventanas completas, factor de reduccion 3 y
media recortada por defecto. Pueden configurarse esas reglas, pero no bajar el
minimo a menos de cuatro ni activar poda semantica en DL: el motor actual aplica
la poda semantica desde la primera ventana. Desactivarla aqui respeta el minimo
sin modificar el motor ni crear un podador propio. ASHA sigue siendo el unico
asignador de recursos; no se podan epocas dentro de un ajuste.

Los errores de entrenamiento quedan bajo el manejo de fallos del Walk-Forward
compartido. El estudio conserva `completado`, `podado` y `fallido` por separado;
no se convierten excepciones de entrenamiento en prunes artificiales.

### Reanudacion

El selector acepta `raiz_corrida` y `run_id`; la estrategia acepta `raiz_corrida`
y exige una `sesion` estable. La estrategia deriva el identificador de corrida
de un hash del SKU y la sesion, sin interpretar esos textos como rutas.
Todo checkpoint, estado e incompatibilidad pertenece al controlador ya existente
en `optimizacion/control_reanudacion`. No se agregan controladores DL ni se usa
el stub de `forecasting`. Una corrida completada se reconstruye sin entrenar;
cambiar datos, parametros temporales o espacio/cota bajo la misma identidad se
rechaza por la huella compartida.

## Pruebas

```sh
uv run pytest tests/comun/modelos/test_mlp.py tests/optimizacion/modelos_deep_learning
```

Incluyen gradientes por diferencias finitas, aprendizaje de una serie periodica,
semillas/dropout/refit, rangos y costo, una seleccion real pequena con TPE y
Walk-Forward, prefijos causales, estados de poda/fallo, persistencia e interrupcion,
y despacho del router para las cuatro clases sin modificar la matriz.

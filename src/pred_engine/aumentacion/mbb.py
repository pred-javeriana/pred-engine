import numpy as np
"""
Funcion: moving_block_bootstrap
Argumentos: 
- serie_sku: array con el residuo de la serie de tiempo.
- block_size: tamaño del bloque que se va a utilizar ej. 3 indica usar bloques de 3 datos
- tamaño_esperado: tamaño final que se quiere ej. para pasar de 500 a 750 es 750
Return:
- el array luego del proceso con el data augmentation
Funcionamiento: la funcion toma el array dado, lo divide en blocks
y al azar elige los bloques necesarios para llegar al tamaño_esperado
"""
def moving_block_bootstrap(
    serie_sku: np.ndarray,
    block_size: int,
    tamaño_esperado: int
) -> np.ndarray:

# Validación de la entrada

    #Validacion de dimensiones
    if serie_sku.ndim != 1:
        printf("serie_sku debe ser un array 1D.")#esto debe ser un log con logger

    #Validacion de vacio
    if serie_sku.size == 0:
        printf("serie_sku no puede estar vacío.") #esto debe ser un log con logger
    
    #Validaciones bloque valido
    if block_size <= 0:
        printf("block_size debe ser mayor que 0.")#esto debe ser un log con logger

    if block_size > serie_sku.size:
        printf("block_size no puede ser mayor que la longitud de la serie.")#esto debe ser un log con logger

    #Validaciones tamaño final
    if tamaño_esperado< 0:
        printf("tamaño_esperado no puede ser negativo.")#esto debe ser un log con logger

    if tamaño_esperado == 0:
        return np.empty(0, dtype=serie_sku.dtype) #reportar en logger y mostrar error en pantalla

    #generador random
    rng = np.random.default_rng(42)

    n = serie_sku.size

    #Cantidad de bloques necesarios
    num_blocks = int(np.ceil(tamaño_esperado / block_size))

    #inicio maximo de bloque (de 0 a max_start)
    max_start = n - block_size

    i_bloques = rng.integers(
        0,
        max_start + 1,
        size=num_blocks
    )

    # Extraer y concatenar los bloques
    bloques = [
        serie_sku[i:i + block_size]
        for i in i_bloques
    ]

    serie_aumentada = np.concatenate(bloques)

    #Corta el array en tamaño esperado
    serie_aumentada = serie_aumentada[:tamaño_esperado]

    return serie_aumentada
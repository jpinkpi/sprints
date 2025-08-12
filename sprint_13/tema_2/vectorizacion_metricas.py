#Vectorización de métricas
'''
Vamos a escribir funciones para la evaluación de métricas mediante vectores.
¿Cómo podemos aplicar la vectorización a las métricas de evaluación? Veamos un ejemplo. 
Almacena un conjunto de valores reales en la variable target y valores pronosticados en la variable predictions. Ambos conjuntos son de tipo np.array.'''

import numpy as np

target = np.array([0.9, 1.2, 1.4, 1.5, 1.9, 2.0])
predictions = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

"""
Utiliza los métodos estándar de NumPy para calcular las métricas de evaluación:

- sum() — para encontrar la suma de los elementos de una matriz
- mean() — para calcular el valor medio

Llámalas de la siguiente manera: <array name>.sum() y <array name>.mean().

Por ejemplo, esta es la fórmula para calcular el error cuadrático medio (ECM):

    ECM = (1 / n) * Σ_{i=1}^n (target_i − predictions_i)^2

donde:
- n es la longitud de cada matriz
- Σ es la suma de todas las observaciones de la muestra (i varía de 1 a n)
- Los elementos ordinales de los vectores target y predictions se denotan por target_i y predictions_i.

Escribe la fórmula utilizando sum():
"""

def mse1(target, predictions):
    n = target.size
    return ((target - predictions) ** 2).sum() / n

"Vamos a escribir la fórmula de ECM utilizando mean():"
def mse2(target, predictions):
    return ((target - predictions) ** 2).mean()

'Para asegurarnos de que los resultados coincidan, apliquemos las dos funciones ECM a las matrices target y predictions.'
print(mse1(target, predictions), mse2(target, predictions))

#EJERCICIO 1
"""
Escribe la función para calcular el EAM utilizando mean(). Encuentra el EAM y muestra los resultados (en precódigo).

EAM = (1 / n) * Σ_{i=1}^n |target_i - predictions_i|

donde:
- n es la longitud de cada matriz
- Σ es la suma de todas las observaciones de la muestra
- Los elementos ordinales de los vectores target y predictions se denotan por target_i y predictions_i
"""
import numpy as np


def mae(target, predictions):
    return np.abs(target - predictions).mean() # < escribe tu código aquí >


target = np.array([0.9, 1.2, 1.4, 1.5, 1.9, 2.0])
predictions = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
print(mae(target, predictions))

#EJERCICIO 2
"""
Calcula el RECM mediante la fórmula:

RECM = ECM = (1 / n) * Σ_{i=1}^n (target_i - predictions_i)^2

donde:
- n es la longitud de cada matriz
- Σ es la suma de todas las observaciones de la muestra
- Los elementos ordinales de los vectores target y predictions se denotan por target_i y predictions_i

Muestra los resultados (en precódigo).
"""

import numpy as np


def rmse(target, predictions):
    return np.sqrt(((target-predictions)**2).mean()) # < escribe tu código aquí >


target = np.array([0.9, 1.2, 1.4, 1.5, 1.9, 2.0])
predictions = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
print(rmse(target, predictions))

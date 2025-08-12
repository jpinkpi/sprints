
#Funciones vectorizadas
"""
Las funciones pueden escribirse utilizando vectores. A este tipo se le denomina funciones vectorizadas. Nos ayudan a operar sobre un vector completo de una manera eficiente.
Las herramientas de NumPy nos permiten realizar varias operaciones con vectores. Si utilizamos el método np.array() para crear un vector, podemos multiplicar y dividir dos matrices del mismo tamaño (esta multiplicación y división que discutiremos a continuación, son elemento a elemento. Existen otros tipos de multiplicación y división de matrices, pero las discutiremos más adelante.). Como resultado, obtendremos un nuevo vector del mismo tamaño:
"""

import numpy as np

array1 = np.array([2, -4, 6, -8])
array2 = np.array([1, 2, 3, 4])
array_mult = array1 * array2
array_div = array1 / array2
print('Producto de dos matrices: ', array_mult)
print('Cociente de dos matrices: ', array_div)

# Producto de dos matrices:  [  2  -8  18 -32]
# Cociente de dos matrices:  [ 2. -2.  2. -2.]

"""
Si las operaciones aritméticas se realizan sobre una matriz y un solo número, la acción se aplica a cada elemento de la matriz. Y de nuevo, se forma una matriz del mismo tamaño.

Vamos a ver cómo se verá si realizamos sumas, restas y divisiones sobre una matriz con un escalar:
"""

array2 = np.array([1, 2, 3, 4])
array2_plus_10 = array2 + 10
array2_minus_10 = array2 - 10
array2_div_10 = array2 / 10
print('Suma: ', array2_plus_10)
print('Resta:', array2_minus_10)
print('Cociente:', array2_div_10)

# Suma:  [11 12 13 14]
# Resta:  [-9 -8 -7 -6]
# Cociente:  [0.1 0.2 0.3 0.4]

"""
El mismo principio de "elemento por elemento" se aplica a las matrices cuando tratamos con operaciones matemáticas estándar como las de exponentes o logaritmos.

Vamos a elevar una matriz a la segunda potencia:
"""

numbers_from_0 = np.array([0, 1, 2, 3, 4])
squares = numbers_from_0 ** 2
print(squares)

# [ 0  1  4  9 16]

"""
Todo esto también lo podemos hacer con listas a través de bucles, pero las operaciones con vectores en NumPy son mucho más rápidas. Uno de los modos más útiles de aprovechar esta capacidad de las matrices NumPy es escribir funciones. Podemos pasar matrices como entradas a las funciones al igual que lo hacemos con otros tipos de datos.

Aquí tenemos un ejemplo: hay una matriz de valores cuyos datos deben ser escalados a efectos de nuestro análisis. Cada elemento de la matriz debe ser escalado a un número que oscile entre 0 y 1, donde 0 es el valor mínimo de la matriz y 1 el valor máximo. Aquí está la fórmula que podemos utilizar para este tipo de función:

f(x) = (x - MIN) / (MAX - MIN)

Para aplicar esta función a todos los elementos de la matriz, llamamos a las funciones max() y min() a fin de encontrar sus valores máximos y mínimos. Una vez definida nuestra función, podemos pasarle una matriz de números. Como resultado, obtenemos una matriz de la misma longitud, pero con elementos convertidos:
"""

def min_max_scale(values):
    return (values - min(values)) / (max(values) - min(values))

our_values = np.array([-20, 0, 0.5, 80, -1])
print(min_max_scale(our_values))

# [0.    0.2   0.205 1.    0.19 ]

"""
A veces los valores son arbitrariamente grandes. En estos casos, una función de escalado basada en los valores mínimo y máximo no será efectiva. Podemos resolver este problema utilizando la función logística o la transformación logística:

f(x) = 1 / (1 + exp(-x))

donde exp() es la función exponente (del lat. exponere, "exponer").
Eleva e, el número de Euler, a la potencia del argumento. Este número recibió el nombre del gran matemático suizo Leonhard Euler y es aproximadamente igual a 2.718281828.
"""

#EJERCICIO
'''
Escribe la función logistic_transform() para realizar la transformación logística. 
Aplícala a todos los elementos de la matriz. Muestra los resultados (en precódigo).
'''
import numpy as np


def logistic_transform(values):
    return  1 / (1 + (np.exp(-values)))# < escribe tu código aquí >


our_values = np.array([-20, 0, 0.5, 80, -1])
print(logistic_transform(our_values))
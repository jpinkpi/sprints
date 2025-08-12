#OPERACIONES CON ELEMENTOS DE MATRIZ
import numpy as np

'''
OPERACIONES CON ELEMENTOS DE MATRIZ

Puedes realizar las mismas operaciones con elementos de matriz que con elementos vectoriales.
Estas operaciones matriciales serán útiles cuando se combinen múltiples matrices en una sola matriz.
Las matrices se pueden sumar, restar o multiplicar.

Considera que estas operaciones se realizan **elemento por elemento**, como se muestra a continuación, 
y como resultado, las matrices en las operaciones deben ser del mismo tamaño.
El resultado de cualquiera de estas operaciones es una matriz del mismo tamaño.
'''

'''
SUMA DE MATRICES

Intentemos sumar dos matrices.
'''
matrix1 = np.array([
    [1, 2], 
    [3, 4]])

matrix2 = np.array([
    [5, 6], 
    [7, 8]])

print("Suma de matrices:")
print(matrix1 + matrix2)
# Resultado:
# [[ 6  8]
#  [10 12]]

'''
RESTA DE MATRICES

Puedes restar una matriz de otra como se muestra:
'''
matrix3 = np.array([
    [5, 10], 
    [1, 20]])

matrix4 = np.array([
    [1, 3], 
    [3, 10]])

print("\nResta de matrices:")
print(matrix3 - matrix4)
# Resultado:
# [[ 4  7]
#  [-2 10]]

'''
MULTIPLICACIÓN ELEMENTO A ELEMENTO

Puedes multiplicar una matriz por otra matriz usando np.multiply():
'''
matrix5 = np.array([
    [1, 2], 
    [3, 4]])

matrix6 = np.array([
    [5, 6], 
    [7, 8]])

print("\nMultiplicación elemento a elemento:")
print(np.multiply(matrix5, matrix6))
# Resultado:
# [[ 5 12]
#  [21 32]]

'''
MULTIPLICACIÓN, SUMA Y RESTA ESCALAR

También podemos multiplicar, sumar o restar un número a cada elemento de una matriz.
'''
matrix7 = np.array([
    [1, 2], 
    [3, 4]])

print("\nMultiplicación por escalar (2):")
print(matrix7 * 2)
# Resultado:
# [[2 4]
#  [6 8]]

print("\nSuma de escalar (2):")
print(matrix7 + 2)
# Resultado:
# [[3 4]
#  [5 6]]

print("\nResta de escalar (2):")
print(matrix7 - 2)
# Resultado:
# [[-1  0]
#  [ 1  2]]

'''
DIVISIÓN ENTRE MATRICES

No se puede hacer una división entre matrices directamente como las otras operaciones.
Más adelante se verá una forma análoga con la matriz inversa.
'''



#Ejercicio 1
'''
El operador de telefonía móvil T-phone está de vuelta con los datos de los usuarios de toda la semana. 
Unifica los datos en una matriz para la semana: monday (lunes), tuesday (martes), wednesday (miércoles), thursday (jueves),
 friday (viernes), saturday (sábado), sunday (domingo). Muéstrala en la pantalla (en precódigo).

Crea un pronóstico para el mes. Con base en datos semanales, averigüa cuánto usa cada cliente los servicios en promedio por día. 
Luego multiplica este promedio por 30.4 (el número promedio de días en un mes).
'''

import numpy as np
import pandas as pd

services = ['Minutos', 'Mensajes', 'Megabytes']

monday = np.array([
    [10, 2, 72],
    [3, 5, 111],
    [15, 3, 50],
    [27, 0, 76],
    [7, 1, 85]])

tuesday = np.array([
    [33, 0, 108],
    [21, 5, 70],
    [15, 2, 15],
    [29, 6, 34],
    [2, 1, 146]])

wednesday = np.array([
    [16, 0, 20],
    [23, 5, 34],
    [5, 0, 159],
    [35, 1, 74],
    [5, 0, 15]])

thursday = np.array([
    [25, 1, 53],
    [15, 0, 26],
    [10, 0, 73],
    [18, 1, 24],
    [2, 2, 24]])

friday = np.array([
    [32, 0, 33],
    [4, 0, 135],
    [2, 2, 21],
    [18, 5, 56],
    [27, 2, 21]])

saturday = np.array([
    [28, 0, 123],
    [16, 5, 165],
    [10, 1, 12],
    [42, 4, 80],
    [18, 2, 20]])

sunday = np.array([
    [30, 4, 243],
    [18, 2, 23],
    [12, 2, 18],
    [23, 0, 65],
    [34, 0, 90]])

weekly = monday + tuesday + wednesday + thursday + friday + saturday + sunday # < escribe tu código aquí >

print('En una semana')
print(pd.DataFrame(weekly, columns=services))
print()

forecast = (weekly * 30.4)/7  # < escribe tu código aquí >

print('Pronóstico para un mes')
print(pd.DataFrame(forecast, dtype=int, columns=services))
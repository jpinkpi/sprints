#Transposición de matriz

import numpy as np
import pandas as pd

'''
TRANSPOSICIÓN DE MATRIZ

A veces es útil intercambiar las columnas y las filas de una matriz.
Esto da como resultado una nueva matriz llamada transpuesta.

Para transponer una matriz, la "volteamos" sobre su diagonal principal (de esquina superior izquierda a inferior derecha).
Esto convierte una matriz de tamaño m×n en una de tamaño n×m. 
Es decir: las filas se convierten en columnas, y las columnas en filas.

Se representa con el símbolo T:  Aᵀ
'''

# Crear una matriz
matrix = np.array([
    [1, 2], 
    [4, -4], 
    [0, 17]
])

print("Matriz original:")
print(matrix)

print("\nMatriz transpuesta:")
print(matrix.T)

'''
Matriz original:
[[ 1  2]
 [ 4 -4]
 [ 0 17]]

Matriz transpuesta:
[[ 1  4  0]
 [ 2 -4 17]]

Esta operación es muy útil, por ejemplo, en machine learning para reorganizar ecuaciones,
convertir observaciones en vectores, o adaptar matrices para multiplicaciones.
'''

# Multiplicación fallida con vector mal dimensionado
vector = [2, 1, -1]

print("\nIntento de multiplicar matriz original por vector (dimensiones incompatibles):")
try:
    print(np.dot(matrix, vector))
except ValueError as e:
    print("Error:", e)

'''
Resultado:
Error: shapes (3,2) and (3,) not aligned: 2 (dim 1) != 3 (dim 0)

La matriz original es de forma (3x2), y el vector tiene tamaño (3,). 
No es posible multiplicarlos porque el número de columnas de la matriz no coincide con la longitud del vector.
Pero podemos transponer la matriz para hacerlo funcionar:
'''

print("\nMultiplicación después de transponer la matriz:")
print(np.dot(matrix.T, vector))
# Resultado: [  6 -17]

'''
EXPLICACIÓN: 
- matrix.T es de forma (2x3)
- vector es de forma (3,)
→ producto válido → resultado de forma (2,)

Ahora veamos un ejemplo real.
Una fábrica (Roble y Pino) produce muebles hechos con tres materiales: tablas de madera, tubos de metal y tornillos.

Para producir una silla, banco y mesa, se necesita:

Elemento     Tabla (m)   Tubos (m)   Tornillos (qt)
Silla           0.2         1.2           8
Banco           0.5         0.8           6
Mesa            0.8         1.6           8
'''

# Crear matriz de materiales por mueble (filas: muebles, columnas: materiales)
products = ['Silla', 'Banco', 'Mesa']
materials = ['Tabla (m)', 'Tubos (m)', 'Tornillos (qt)']

resources = np.array([
    [0.2, 1.2, 8],   # Silla
    [0.5, 0.8, 6],   # Banco
    [0.8, 1.6, 8]    # Mesa
])

print("\nRecursos necesarios por tipo de mueble:")
print(pd.DataFrame(resources, index=products, columns=materials))

'''
Ahora, si queremos reorganizar la información para tener los materiales como filas y los productos como columnas,
simplemente transponemos la matriz:
'''

print("\nMatriz transpuesta (materiales como filas):")
print(pd.DataFrame(resources.T, index=materials, columns=products))

'''
Esto es útil, por ejemplo, para calcular cuántos recursos se necesitan si se fabrican ciertas cantidades de cada mueble.

Ejemplo: fabricar [10 sillas, 5 bancos, 2 mesas]
'''

cantidad = np.array([10, 5, 2])  # vector con cantidades de muebles

total_materiales = np.dot(resources.T, cantidad)

print("\nTotal de materiales requeridos para 10 sillas, 5 bancos y 2 mesas:")
print(pd.Series(total_materiales, index=materials))

'''
Resultado:
Tabla (m)        6.6
Tubos (m)       16.4
Tornillos (qt) 126.0
'''

#Ejercicio 1
#Calcula cuánto material se necesitará para producir 16 mesas, 60 sillas y 4 bancos. Guarda el resultado en la variable de materiales y muestra el resultado en la pantalla (en precódigo).
import numpy  as np
import pandas as pd

materials_names = ['Tabla', 'Tubos', 'Tornillos']

manufacture = np.array([
    [0.2, 1.2, 8],
    [0.5, 0.8, 6],
    [0.8, 1.6, 8]])

furniture = [60, 4, 16]

materials = np.dot(manufacture.T, furniture)# < escribe tu código aquí >

print(pd.DataFrame([materials], index=[''], columns=materials_names))
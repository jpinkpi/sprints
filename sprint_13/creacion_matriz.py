#Creación de matrices
"Creación de matrices"
"""

"Primero, veamos qué es una matriz y en qué se diferencia de un vector.  
Una matriz es una tabla rectangular o tabla bidimensional que consta de m filas y n columnas (el tamaño se escribe como 𝑚×𝑛).  
Las matrices generalmente se denotan con letras latinas mayúsculas, por ejemplo, A.  
Sus elementos están en minúsculas con doble índice a_ij, donde i es el número de fila y j es el número de columna."

"Puedes pensar en una matriz como filas de vectores o columnas de vectores, donde cada vector es una lista de números que pueden estar en una fila o en una columna."

"Supongamos que la matriz A contiene dos filas y tres columnas. Sus elementos son:  
a₁₁ = 1, a₁₂ = 2, a₁₃ = 3,  
a₂₁ = 2, a₂₂ = 3 y a₂₃ = 4."

"A = [[1, 2, 3],  
       [2, 3, 4]]"

"Puedes crear una matriz en Python usando la función np.array().  
Todo lo que necesitas es una lista de listas que tengan la misma longitud.  
Creemos una matriz de 3×3 con tres vectores iguales [1, 2, 3], [4, 5, 6] y [7, 8, 9]."
"""
import numpy as np

matrix = np.array([
    [1, 2, 3], 
    [4, 5, 6],
    [7, 8, 9]
])
print(matrix)

[[1 2 3]
 [4 5 6]
 [7 8 9]]

"💡Nota: En Python, este código:"

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

"equivale a:"

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

""""
"Ambos producen el mismo resultado; la diferencia tan solo se encuentra en el formato.  
La primera versión está espaciada para facilitar la lectura, lo cual ayuda a comprender la estructura de la matriz."

"Alternativamente, también podemos crear una matriz a partir de vectores.  
Usemos vector1 y vector2 para crear una matriz de 2×3.  
Considera que nuestros vectores siempre deben tener la misma longitud."
"""

vector1 = np.array([1, 2, 3])
vector2 = np.array([-1, -2, -3])

list_of_vectors = [vector1, vector2]

matrix_from_vectors = np.array(list_of_vectors)

print(matrix_from_vectors)

[[ 1  2  3]
 [-1 -2 -3]]

""""
"Ahora veamos otra forma de crear una matriz.  
Crearemos una a partir de una tabla de pandas, usando el atributo `values`."

"Presta atención: esta es una herramienta importante que usarás mientras entrenas tu modelo.  
Este paso garantiza que el dataset esté en el formato adecuado para su uso."
"""


import pandas as pd

df = pd.DataFrame({'a': [120, 60, 75], 'b': [42, 50, 90]})
matrix = df.values
print(df)
print()
print(matrix)


"""
"Para determinar el tamaño de una matriz, usamos el atributo `shape`.  
Esto nos da una salida en el formato (filas, columnas).  
Por ejemplo, el siguiente código nos dice que tenemos una matriz de 2×3."
"""


A = np.array([
    [1, 2, 3], 
    [2, 3, 4]])

print('Tamaño:', A.shape) 

"""
"Una operación útil que podríamos necesitar realizar es acceder a un elemento de la matriz.  
Usando la matriz A en el código anterior, el elemento a_ij se llama con `A[i, j]`, donde `i` es la fila y `j` la columna, enumeradas desde cero."

"Veamos esto en el siguiente ejemplo donde recuperamos el elemento de valor 4:"
"""

print('A[1, 2]:', A[1, 2])


"""
"Otra operación que podemos realizar en matrices es seleccionar filas y columnas individuales.  
Podemos usar esta operación para seleccionar una variable objetivo y sus características mientras preparamos un modelo, por ejemplo."

"Vamos a seleccionar la primera fila y la tercera columna de nuestra matriz.  
Podemos seleccionar la fila y la columna como sería de esperar, pero presta atención a cómo usamos `:` para acceder a todos los valores en la fila o columna respectiva."
"""
matrix = np.array([
    [1, 2, 3], 
    [4, 5, 6],
    [7, 8, 9],
    [10,11,12]])

print('Primera fila:', matrix[0, :])
print('Tercera columna:', matrix[:, 2])

"""
"Esta operación da como resultado un vector.  
Sin embargo, también podemos seleccionar varias filas y columnas, es decir, seleccionar una matriz de 2×2 de una matriz de 4×3."
"""

#Ejercicio 1 

'A partir de una lista de filas, crea una matriz de 2×3 donde la primera fila contenga los números 3, 14, 159 y la segunda -2, 7, 183.'

import numpy as np

matrix = np.array([[3,14,159], [-2,7,183]]) # < escribe tu código aquí >

print(matrix)

#Ejercicio 2
'''
El operador de telefonía móvil T-phone quiere analizar los datos de uso diario de los clientes. 
La tabla muestra la cantidad de mensajes de texto, minutos de conversación y megabytes de cinco clientes en el día lunes (monday). 
Convierte la tabla en una matriz y muéstrala en la pantalla (en precódigo).
'''

import numpy as np
import pandas as pd

monday_df = pd.DataFrame({'Minutos': [10, 3, 15, 27, 7], 
                          'Mensajes': [2, 5, 3, 0, 1], 
                          'Megabytes': [72, 111, 50, 76, 85]})
 
monday =np.array(monday_df) # < escribe tu código aquí >

print(monday)

#Ejercicio 3
'Selecciona los datos para el cuarto cliente de la matriz resultante (¡cuidado con los índices!). Muestra el resultado en la pantalla.'

import numpy as np
import pandas as pd

monday_df = pd.DataFrame({'Minutes': [10, 3, 15, 27, 7], 
                          'Messages': [2, 5, 3, 0, 1], 
                          'Megabytes': [72, 111, 50, 76, 85]}) 
 
monday = monday_df.values

print(monday[3,:])# < escribe el código aquí >)

#Ejercicio 4
'En la matriz monday, selecciona la cantidad de datos que utilizó cada cliente. Muestra el resultado en la pantalla.'
import numpy as np
import pandas as pd

monday_df = pd.DataFrame({'Minutes': [10, 3, 15, 27, 7], 
                          'Messages': [2, 5, 3, 0, 1], 
                          'Megabytes': [72, 111, 50, 76, 85]})
 
monday = monday_df.values

print(monday[:,2])# < escribe el código aquí >)
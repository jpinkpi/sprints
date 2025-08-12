#Suma y resta de vectores


"""
Podemos sumar y restar vectores del mismo tamaño.
Al sumar o restar vectores, se hace la operación para cada elemento de estos.
Para sumar dos vectores, hay que sumar las coordenadas correspondientes de los vectores iniciales.
A la hora de restar, cada coordenada del vector resultante es igual a la diferencia entre coordenadas de los vectores iniciales.

Vector        Coordenadas
x⃗            (x1, x2, ..., xn)
y⃗            (y1, y2, ..., yn)
x⃗ + y⃗       (x1 + y1, x2 + y2, ..., xn + yn)
x⃗ -y⃗       (x1 - y1, x2 - y2, ..., xn - yn)

Ejemplo:
Suma el vector1 — [2, 3] y el vector2 — [6, 2]
"""

vector1 = [2, 3]
vector2 = [6, 2]

suma = [x + y for x, y in zip(vector1, vector2)]

print("Suma:", suma)

'Resta el vector1 del vector2:'
subtraction = vector2 - vector1
print(subtraction)

'Traza con flechas en el plano los vectores obtenidos:'
import numpy as np
import matplotlib.pyplot as plt

vector1 = np.array([2, 3])
vector2 = np.array([6, 2])
sum_of_vectors = vector1 + vector2
subtraction = vector2 - vector1

plt.figure(figsize=(10.2, 10))
plt.axis([0, 8.4, -2, 6])
arrow1 = plt.arrow(
    0,
    0,
    vector1[0],
    vector1[1],
    head_width=0.3,
    length_includes_head="True",
    color='b',
)
arrow2 = plt.arrow(
    0,
    0,
    vector2[0],
    vector2[1],
    head_width=0.3,
    length_includes_head="True",
    color='g',
)
arrow3 = plt.arrow(
    0,
    0,
    sum_of_vectors[0],
    sum_of_vectors[1],
    head_width=0.3,
    length_includes_head="True",
    color='r',
)
arrow4 = plt.arrow(
    0,
    0,
    subtraction[0],
    subtraction[1],
    head_width=0.3,
    length_includes_head="True",
    color='m',
)
plt.plot(0, 0, 'ro')
plt.legend(
    [arrow1, arrow2, arrow2, arrow3],
    ['vector1', 'vector2', 'sum_of_vectors', 'subtraction'],
    loc='upper left',
)
plt.grid(True)
plt.show()

'''Si trazamos un vector que sea igual al vector1 azul en términos de longitud y 
dirección desde el final del vector2 verde, obtendremos el vector rojo (sum_of_vectors).'''

import numpy as np
import matplotlib.pyplot as plt

vector1 = np.array([2, 3])
vector2 = np.array([6, 2])
sum_of_vectors = vector1 + vector2

plt.figure(figsize=(10.2, 10))
plt.axis([0, 8.4, -2, 6])
arrow1 = plt.arrow(
    0,
    0,
    vector1[0],
    vector1[1],
    head_width=0.3,
    length_includes_head="True",
    color='b',
)
arrow2 = plt.arrow(
    0,
    0,
    vector2[0],
    vector2[1],
    head_width=0.3,
    length_includes_head="True",
    color='g',
)
arrow2new = plt.arrow(
    vector1[0],
    vector1[1],
    vector2[0],
    vector2[1],
    head_width=0.3,
    length_includes_head="True",
    color='g',
)
arrow3 = plt.arrow(
    0,
    0,
    sum_of_vectors[0],
    sum_of_vectors[1],
    head_width=0.3,
    length_includes_head="True",
    color='r',
)
plt.plot(0, 0, 'ro')
plt.legend(
    [arrow1, arrow2, arrow3],
    ['vector1', 'vector2', 'sum_of_vectors'],
    loc='upper left',
)
plt.grid(True)
plt.show()


'''
El triángulo obtenido en el gráfico anterior nos da el sentido geométrico de la suma de vectores.
Si cada vector es un movimiento en una dirección determinada, 
la suma de dos vectores es el movimiento a lo largo del primer vector seguido del movimiento a lo largo del segundo.
La diferencia entre dos vectores es un paso, por ejemplo, a lo largo del vector2, 
seguido de un paso en la dirección opuesta al vector1.
Traza el vector de substraction:
'''

import numpy as np
import matplotlib.pyplot as plt

vector1 = np.array([2, 3])
vector2 = np.array([6, 2])
sum_of_vectors = vector1 + vector2
subtraction = vector2 - vector1

plt.figure(figsize=(10.2, 10))
plt.axis([0, 8.4, -2, 6])
arrow1 = plt.arrow(
    0,
    0,
    vector1[0],
    vector1[1],
    head_width=0.3,
    length_includes_head="True",
    color='b',
)
arrow2 = plt.arrow(
    0,
    0,
    vector2[0],
    vector2[1],
    head_width=0.3,
    length_includes_head="True",
    color='g',
)
arrow4 = plt.arrow(
    0,
    0,
    subtraction[0],
    subtraction[1],
    head_width=0.3,
    length_includes_head="True",
    color='m',
)
arrow1new = plt.arrow(
    vector2[0],
    vector2[1],
    -vector1[0],
    -vector1[1],
    head_width=0.3,
    length_includes_head="True",
    color='b',
)
plt.plot(0, 0, 'ro')
plt.legend(
    [arrow1, arrow2, arrow4],
    ['vector1', 'vector2', 'subtraction'],
    loc='upper left',
)
plt.grid(True)
plt.show()

'''
Aquí tenemos un ejemplo: dos tiendas online con un surtido idéntico de productos, 
FoCase y Caseology, están planeando una fusión. 
Las tablas contienen fragmentos de sus listas de existencias. 
Las columnas representan los nombres de los productos y las cantidades.
'''

#Consulta las existencias de FoCase:
import numpy as np
import pandas as pd

quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = [
    'Capa de silicone para o iPhone 8',
    'Capa de couro para o iPhone 8',
    'Capa de silicone para o iPhone XS',
    'Capa de couro para o iPhone XS',
    'Capa de silicone para o iPhone XS Max',
    'Capa de couro para o iPhone XS Max',
    'Capa de silicone para o iPhone 11',
    'Capa de couro para o iPhone 11',
    'Capa de silicone para o iPhone 11 Pro',
    'Capa de couro para o iPhone 11 Pro',
]
stocks_1 = pd.DataFrame({'Quantity': quantity_1}, index=models)
print(stocks_1)


#Consulta las existencias de Caseology:
quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Quantity': quantity_2}, index=models)
print(stocks_2)

#EJERCICIO 1
"Toma la columna de cantidades de cada tabla y conviértela en un vector. Muestra los resultados (en precódigo)."

import numpy as np
import pandas as pd

quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = [
    'Capa de silicone para o iPhone 8',
    'Capa de couro para o iPhone 8',
    'Capa de silicone para o iPhone XS',
    'Capa de couro para o iPhone XS',
    'Capa de silicone para o iPhone XS Max',
    'Capa de couro para o iPhone XS Max',
    'Capa de silicone para o iPhone 11',
    'Capa de couro para o iPhone 11',
    'Capa de silicone para o iPhone 11 Pro',
    'Capa de couro para o iPhone 11 Pro',
]
stocks_1 = pd.DataFrame({'Quantity': quantity_1}, index=models)
quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Quantity': quantity_2}, index=models)

vector_of_quantity_1 =  stocks_1["Quantity"].values # < escribe el código aquí  >
vector_of_quantity_2 =  stocks_2["Quantity"].values# < escribe el código aquí >
print(
    'Existencias de la primera tienda:',
    vector_of_quantity_1,
    '\nExistencias de la segunda tienda:',
    vector_of_quantity_2,
)

#EJERCICIO 2
"Encuentra el vector de existencias de FoCaseology después de la fusión. Muestra los resultados (en precódigo)."

import numpy as np
import pandas as pd

quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = [
    'Capa de silicone para o iPhone 8',
    'Capa de couro para o iPhone 8',
    'Capa de silicone para o iPhone XS',
    'Capa de couro para o iPhone XS',
    'Capa de silicone para o iPhone XS Max',
    'Capa de couro para o iPhone XS Max',
    'Capa de silicone para o iPhone 11',
    'Capa de couro para o iPhone 11',
    'Capa de silicone para o iPhone 11 Pro',
    'Capa de couro para o iPhone 11 Pro',
]
stocks_1 = pd.DataFrame({'Quantity' : quantity_1}, index=models)
quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Quantity' : quantity_2}, index=models)

vector_of_quantity_1 = stocks_1['Quantity'].values
vector_of_quantity_2 = stocks_2['Quantity'].values

vector_of_quantity_united = vector_of_quantity_1 + vector_of_quantity_2# < escribe el código aquí >
print(vector_of_quantity_united)


#EJERCICIO 3
'Crea un DataFrame para el inventario de la tienda después de la fusión. Muestra los resultados (en precódigo).'

import numpy as np
import pandas as pd

quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = [
    'Capa de silicone para o iPhone 8',
    'Capa de couro para o iPhone 8',
    'Capa de silicone para o iPhone XS',
    'Capa de couro para o iPhone XS',
    'Capa de silicone para o iPhone XS Max',
    'Capa de couro para o iPhone XS Max',
    'Capa de silicone para o iPhone 11',
    'Capa de couro para o iPhone 11',
    'Capa de silicone para o iPhone 11 Pro',
    'Capa de couro para o iPhone 11 Pro',
]
stocks_1 = pd.DataFrame({'Quantity' : quantity_1}, index=models)
quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Quantity' : quantity_2}, index=models)

vector_of_quantity_1 = stocks_1['Quantity'].values
vector_of_quantity_2 = stocks_2['Quantity'].values
vector_of_quantity_united = vector_of_quantity_1 + vector_of_quantity_2

stocks_united = pd.DataFrame({'Quantity' : vector_of_quantity_united}, index=models # < escribe tu código aquí >
                            )
print(stocks_united)



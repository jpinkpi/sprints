#Valor medio de los vectores
"""
Vamos a ver cómo se calcula el valor medio de un conjunto de vectores. Este valor representará la media de un conjunto de puntos de datos almacenados en un vector.
Si, por ejemplo, los vectores individuales de un conjunto describen a los clientes en función de sus características, entonces el valor medio de los vectores suele describir a un cliente típico o estadísticamente promedio.

Para el conjunto de vectores  
a1, a2 … an (donde n es el número total de vectores), el valor medio de los vectores es la suma de todos los vectores multiplicada por 1/n. El resultado es un nuevo vector ā:

ā = 1/n (a1 + a2 + ⋯ + an)

Si el conjunto está formado por un solo vector (n=1), será igual a la media: a = a1. 
El valor medio de dos vectores es a = (a1 + a2) / 2. 
El valor medio de un par de vectores bidimensionales es la mitad del segmento que une a1 y a2.

Encuentra la media del vector1 — [2, 3] y del vector2 — [6, 2]:
"""
import numpy as np

vector1 = np.array([2, 3])
vector2 = np.array([6, 2])
vector_mean = 0.5 * (vector1 + vector2)
print(vector_mean)  # [4.  2.5]

"""
Al valor medio lo llamamos vector_mean. 
La primera coordenada del nuevo vector es el valor medio de las primeras coordenadas del vector1 y del vector2, 
y la segunda coordenada es el valor medio de las segundas coordenadas del vector1 y del vector2.

Dibujemos estos vectores en el plano. Traza el vector vector1 + vector2 y multiplícalo por 0,5.
"""
import matplotlib.pyplot as plt

vector_mean = 0.5 * (vector1 + vector2)

plt.figure(figsize=(10, 10))
plt.axis([0, 8.4, -1, 6])
arrow1 = plt.arrow(
    0, 0,
    vector1[0], vector1[1],
    head_width=0.3,
    length_includes_head="True",
    color='b',
)
arrow2 = plt.arrow(
    0, 0,
    vector2[0], vector2[1],
    head_width=0.3,
    length_includes_head="True",
    color='g',
)
arrow_sum = plt.arrow(
    0, 0,
    vector1[0] + vector2[0], vector1[1] + vector2[1],
    head_width=0.3,
    length_includes_head="True",
    color='r',
)
arrow_mean = plt.arrow(
    0, 0,
    vector_mean[0], vector_mean[1],
    head_width=0.3,
    length_includes_head="True",
    color='m',
)
plt.plot(vector1[0], vector1[1], 'ro')
plt.plot(vector2[0], vector2[1], 'ro')
plt.plot(vector_mean[0], vector_mean[1], 'ro')
plt.legend(
    [arrow1, arrow2, arrow_sum, arrow_mean],
    ['vector1', 'vector2', 'vector1+vector2', 'vector_mean'],
    loc='upper left',
)
plt.grid(True)
plt.show()

"""
Intentemos calcular la valoración media de los visitantes de la famosa tienda LuxForVIP.

Para calcular la valoración media del precio, utiliza la función sum(). Esta función encontrará la suma de todos los elementos del vector.
"""
import pandas as pd

ratings_values = [
    [68, 18],
    [81, 19],
    [81, 22],
    [15, 75],
    [75, 15],
    [17, 72],
    [24, 75],
    [21, 91],
    [76, 6],
    [12, 74],
    [18, 83],
    [20, 62],
    [21, 82],
    [21, 79],
    [84, 15],
    [73, 16],
    [88, 25],
    [78, 23],
    [32, 81],
    [77, 35],
]

ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])

price = ratings['Price'].values  # Matriz NumPy con todas las valoraciones de precios
sum_prices = sum(price)  # suma de todas las valoraciones de precios
average_price_rat = sum_prices / len(price)  # valor medio de las valoraciones de precios
print(average_price_rat)  # 49.1

#ejercicio 1
"""
Encuentra la valoración media de la calidad. Muestra los resultados (en precódigo).
"""

import numpy as np
import pandas as pd

ratings_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])
quality = ratings["Quality"].values
sum_qualities = sum(quality)
# < escribe tu código aquí >

average_quality_rat = sum_qualities / len(quality) # < escribe tu código aquí >
print(average_quality_rat)

#ejercicio 2
'Combina las valoraciones medias de calidad y precio de todos los visitantes en un vector. Muestra los resultados (en precódigo).'
import numpy as np
import pandas as pd

ratings_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])

price = ratings['Price'].values
sum_prices = sum(price)
average_price_rat = sum(price) / len(price)

quality = ratings['Quality'].values
average_quality_rat = sum(quality) / len(quality)

average_rat =np.array([average_price_rat, average_quality_rat]) # < escribe tu código aquí >
print(average_rat)

#EJERCICIO 3
'''
Indica el valor obtenido en el plano de coordenadas. ¿Puede considerarse el vector resultante como la valoración media del visitante? 
Muestra los resultados (en precódigo).
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ratings_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])

price = ratings['Price'].values
sum_prices = sum(price)
average_price_rat = sum(price) / len(price)

quality = ratings['Quality'].values
average_quality_rat = sum(quality) / len(quality)
average_rat = np.array([average_price_rat, average_quality_rat])

plt.figure(figsize=(7, 7))
plt.axis([0, 100, 0, 100])
plt.plot(average_rat[0], average_rat[1], 'mo', markersize=15)# <escribe el código aquí >, 'mo', markersize=15)
plt.plot(price, quality, 'ro')
plt.xlabel('Price')
plt.ylabel('Quality')
plt.grid(True)
plt.title('Distribution of ratings and mean value for the whole sample')
plt.show()

#EJERCICIO 4
'''
Calcula por separado las valoraciones medias de los visitantes que proceden del agregador de mercado de masas y de los que proceden del agregador de marcas de lujo. 
Muestra los resultados (en precódigo).
'''

import numpy as np
import pandas as pd

ratings_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])

clients_1 = []
clients_2 = []
for client in list(ratings.values):
    if client[0] < 40 and client[1] > 60:
        clients_1.append(client)
    else:
        clients_2.append(client)

average_client_1 = sum(clients_1) / len(clients_1)
print('Valoración media del primer agregador: ', average_client_1)

# < escribe tu código aquí >
average_client_2 = sum(clients_2) / len(clients_2)
print('Valoración media del segundo agregador: ', average_client_2)

#EJERCICIO 5
'''
Indica los valores obtenidos en el diagrama con las valoraciones individuales de los visitantes.  ¿Puede considerarse cada media obtenida como la valoración media de los visitantes del grupo correspondiente? 
Muestra los resultados (en precódigo).
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ratings_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])
price = ratings['Price'].values
quality = ratings['Quality'].values

clients_1 = []
clients_2 = []
for client in list(ratings.values):
    if client[0] < 40 and client[1] > 60:
        clients_1.append(client)
    else:
        clients_2.append(client)

average_client_1 = sum(clients_1)/len(clients_1)

average_client_2 = sum(clients_2)/len(clients_2)

plt.figure(figsize=(7, 7))
plt.axis([0, 100, 0, 100])

# dibuja la media del grupo 1
# 'b' — azul
plt.plot(average_client_1[0], average_client_1[1],# < escribe el código aquí >, 
         'bo', markersize=15)


# dibuja la media del grupo 2
# 'g' - verde
plt.plot(average_client_2[0], average_client_2[1],# < escribe el código aquí >,
         'go', markersize=15)
plt.plot(price, quality, 'ro')
plt.xlabel('Price')
plt.ylabel('Quality')
plt.grid(True)
plt.title('Distribución de valoraciones y valor medio para cada grupo')
plt.show()
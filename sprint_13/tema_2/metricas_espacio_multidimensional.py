import pandas as pd
from scipy.spatial import distance
import numpy as np


'''
Métricas en el espacio multidimensional
Hasta ahora, solo hemos examinado la distancia entre los puntos de un plano, pero nuestras métricas de distancia también funcionan en el espacio multidimensional.
A la hora de trabajar con los datos para tareas de machine learning, cada observación puede considerarse como un vector.
Por supuesto, como ya sabes, las observaciones con las que trabajamos no suelen ser bidimensionales, más bien contienen información en muchas dimensiones.
vector1 es un vector en cuatro dimensiones:

'''
vector1 = np.array([1, 2, 3, 4])

'''''
El hecho de tener los datos representados en un vector multidimensional nos permite encontrar información significativa utilizando 
la distancia euclidiana como métrica de distancia. Esta métrica de distancia puede ayudarnos a identificar 
el grado de relación entre diferentes observaciones.

Para calcular la distancia euclidiana entre vectores multidimensionales a = (x₁, x₂, …, xₙ) y b = (y₁, y₂, …, yₙ), 
podemos aplicar la misma fórmula de antes, pero con un término extra por cada punto de datos adicional:

        d₂(a, b) = √((x₁ - y₁)² + (x₂ - y₂)² + ⋯ + (xₙ - yₙ)²) 
        = √(∑(xᵢ - yᵢ)²)

La distancia Manhattan también puede calcularse de la siguiente manera:
        d₁(a, b) = |x₁ - y₁| + |x₂ - y₂| + ⋯ + |xₙ - yₙ| 
        = ∑|xᵢ - yᵢ|

Incluso cuando el número de coordenadas es superior a dos, podemos utilizar las conocidas funciones distance.euclidean()
 y distance.cityblock() para calcular distancias en el espacio multidimensional.
Vamos a calcular las distancias euclidiana y Manhattan entre los vectores a = (4, 2, 3, 0, 5) y b = (1, 0, 3, 2, 6).
Podemos ver que tanto nuestro cálculo manual como la función de SciPy no tienen problemas para lidiar con la distancia multidimensional:

'''
import numpy as np
from scipy.spatial import distance

a = np.array([4, 2, 3, 0, 5])
b = np.array([1, 0, 3, 2, 6])

# cálculo manual
d = np.dot(b - a, b - a)**0.5
print(d)

print()
# cálculo con la función euclidiana
e = distance.euclidean(a, b)
print(e)


'Lo mismo ocurre con la distancia Manhattan:'

import numpy as np
from scipy.spatial import distance

a = np.array([4, 2, 3, 0, 5])
b = np.array([1, 0, 3, 2, 6])

d = np.abs(b - a).sum()
print(d)

print()
e = distance.cityblock(a, b)
print(e)

'''
Veamos un ejemplo de la vida real. En el sitio web de la agencia inmobiliaria Cribswithclass.com, 
cada anuncio tiene una serie de parámetros: número de dormitorios, superficie total, tamaño de la cocina, etc. 
Si al cliente le gusta algún apartamento en particular, el sistema de recomendación le ofrecerá opciones similares.

Los datos de los apartamentos se encuentran en la tabla pandas:

'''





columns = [
    'dormitorios',
    'superficie total',
    'cocina',
    'superficie habitable',
    'planta',
    'número de plantas',
]
realty = [
    [1, 38.5, 6.9, 18.9, 3, 5],
    [1, 38.0, 8.5, 19.2, 9, 17],
    [1, 34.7, 10.3, 19.8, 1, 9],
    [1, 45.9, 11.1, 17.5, 11, 23],
    [1, 42.4, 10.0, 19.9, 6, 14],
    [1, 46.0, 10.2, 20.5, 3, 12],
    [2, 77.7, 13.2, 39.3, 3, 17],
    [2, 69.8, 11.1, 31.4, 12, 23],
    [2, 78.2, 19.4, 33.2, 4, 9],
    [2, 55.5, 7.8, 29.6, 1, 25],
    [2, 74.3, 16.0, 34.2, 14, 17],
    [2, 78.3, 12.3, 42.6, 23, 23],
    [2, 74.0, 18.1, 49.0, 8, 9],
    [2, 91.4, 20.1, 60.4, 2, 10],
    [3, 85.0, 17.8, 56.1, 14, 14],
    [3, 79.8, 9.8, 44.8, 9, 10],
    [3, 72.0, 10.2, 37.3, 7, 9],
    [3, 95.3, 11.0, 51.5, 15, 23],
    [3, 69.3, 8.5, 39.3, 4, 9],
    [3, 89.8, 11.2, 58.2, 24, 25],
]

columns = ['bedrooms', 'total area', 'kitchen', 'living area', 'floor', 'total floors']
realty = [
    [1, 38.5, 6.9, 18.9, 3, 5],
    [1, 38.0, 8.5, 19.2, 9, 17],
    [1, 34.7, 10.3, 19.8, 1, 9],
    [1, 45.9, 11.1, 17.5, 11, 23],
    [1, 42.4, 10.0, 19.9, 6, 14],
    [1, 46.0, 10.2, 20.5, 3, 12],
    [2, 77.7, 13.2, 39.3, 3, 17],
    [2, 69.8, 11.1, 31.4, 12, 23],
    [2, 78.2, 19.4, 33.2, 4, 9],
    [2, 55.5, 7.8, 29.6, 1, 25],
    [2, 74.3, 16.0, 34.2, 14, 17],
    [2, 78.3, 12.3, 42.6, 23, 23],
    [2, 74.0, 18.1, 49.0, 8, 9],
    [2, 91.4, 20.1, 60.4, 2, 10],
    [3, 85.0, 17.8, 56.1, 14, 14],
    [3, 79.8, 9.8, 44.8, 9, 10],
    [3, 72.0, 10.2, 37.3, 7, 9],
    [3, 95.3, 11.0, 51.5, 15, 23],
    [3, 69.3, 8.5, 39.3, 4, 9],
    [3, 89.8, 11.2, 58.2, 24, 25],
]

df_realty = pd.DataFrame(realty, columns=columns)
print(df_realty)


#Ejercicio 1
'''
Guarda los vectores de los apartamentos con índices 3 y 11 en las variables vector_first y vector_second. 
Encuentra las distancias euclidiana y Manhattan entre los mismos. Muestra los resultados (en precódigo).
'''

columns = [
    'dormitorios',
    'superficie total',
    'cocina',
    'superficie habitable',
    'planta',
    'número de plantas',
]
realty = [
    [1, 38.5, 6.9, 18.9, 3, 5],
    [1, 38.0, 8.5, 19.2, 9, 17],
    [1, 34.7, 10.3, 19.8, 1, 9],
    [1, 45.9, 11.1, 17.5, 11, 23],
    [1, 42.4, 10.0, 19.9, 6, 14],
    [1, 46.0, 10.2, 20.5, 3, 12],
    [2, 77.7, 13.2, 39.3, 3, 17],
    [2, 69.8, 11.1, 31.4, 12, 23],
    [2, 78.2, 19.4, 33.2, 4, 9],
    [2, 55.5, 7.8, 29.6, 1, 25],
    [2, 74.3, 16.0, 34.2, 14, 17],
    [2, 78.3, 12.3, 42.6, 23, 23],
    [2, 74.0, 18.1, 49.0, 8, 9],
    [2, 91.4, 20.1, 60.4, 2, 10],
    [3, 85.0, 17.8, 56.1, 14, 14],
    [3, 79.8, 9.8, 44.8, 9, 10],
    [3, 72.0, 10.2, 37.3, 7, 9],
    [3, 95.3, 11.0, 51.5, 15, 23],
    [3, 69.3, 8.5, 39.3, 4, 9],
    [3, 89.8, 11.2, 58.2, 24, 25],
]

df_realty = pd.DataFrame(realty, columns=columns)

vector_first =df_realty.iloc[3].values # < escribe tu código aquí >
vector_second = df_realty.iloc[11].values# < escribe tu código aquí >

print('Distancia euclidiana:', distance.euclidean(vector_first, vector_second)) # < escribe tu código aquí >
print('Distancia Manhattan:', distance.cityblock(vector_first, vector_second))# < escribe tu código aquí >

#Ejercicio 2
'''
Supongamos que a un cliente le ha gustado el apartamento con índice 12. 
Encuentra el apartamento más similar basándote en la distancia euclidiana.

Crea una lista con las distancias desde cada vector hasta el vector 12. 
Calcula el índice de la lista más similar y guárdalo en la variable best_index.
'''

columns = [
    'bedrooms',
    'total area',
    'kitchen',
    'living area',
    'floor',
    'total floors',
]
realty = [
    [1, 38.5, 6.9, 18.9, 3, 5],
    [1, 38.0, 8.5, 19.2, 9, 17],
    [1, 34.7, 10.3, 19.8, 1, 9],
    [1, 45.9, 11.1, 17.5, 11, 23],
    [1, 42.4, 10.0, 19.9, 6, 14],
    [1, 46.0, 10.2, 20.5, 3, 12],
    [2, 77.7, 13.2, 39.3, 3, 17],
    [2, 69.8, 11.1, 31.4, 12, 23],
    [2, 78.2, 19.4, 33.2, 4, 9],
    [2, 55.5, 7.8, 29.6, 1, 25],
    [2, 74.3, 16.0, 34.2, 14, 17],
    [2, 78.3, 12.3, 42.6, 23, 23],
    [2, 74.0, 18.1, 49.0, 8, 9],
    [2, 91.4, 20.1, 60.4, 2, 10],
    [3, 85.0, 17.8, 56.1, 14, 14],
    [3, 79.8, 9.8, 44.8, 9, 10],
    [3, 72.0, 10.2, 37.3, 7, 9],
    [3, 95.3, 11.0, 51.5, 15, 23],
    [3, 69.3, 8.5, 39.3, 4, 9],
    [3, 89.8, 11.2, 58.2, 24, 25],
]

df_realty = pd.DataFrame(realty, columns=columns)

# índice del apartamento preferido
preference_index = 12
preference_vector = df_realty.loc[preference_index].values

distances = []
for i in range(df_realty.shape[0]):
    current_vector = df_realty.iloc[i].values
    dist = distance.euclidean(preference_vector, current_vector)# < escribe tu código aquí >
    distances.append(dist)
    
# < escribe tu código aquí >

best_index = np.array(distances).argsort()[1] # < escribe tu código aquí >

print('Índice del apartamento más similar:', best_index)

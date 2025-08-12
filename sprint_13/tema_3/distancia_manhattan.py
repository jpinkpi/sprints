#Distancia Manhattan
"""
Veamos ahora otra métrica de distancia comúnmente utilizada: la distancia Manhattan.
Estaría bien si pudiéramos recorrer todas las distancias en línea recta, como el dron de Flying Stuff de la lección anterior. 
Pero la verdad es que a menudo hay obstáculos que nos obligan a tomar una ruta diferente a la óptima.
Imagina que estás en Nueva York y necesitas ir del punto a al punto b. Lo ideal sería tomar el camino en línea recta (negro), 
que representa la distancia euclidiana. Sin embargo, a menos que tengamos alas, nos veremos obligados a desplazarnos por las manzanas, 
siguiendo las calles y avenidas. 

Estos trayectos se conocen como **distancia Manhattan** o **distancia entre manzanas**. 
Por definición, la distancia Manhattan es la suma de las diferencias absolutas de las 
coordenadas de todos los puntos. Podemos calcularla entre los puntos:

a = (x₁, y₁)  
b = (x₂, y₂)

utilizando la siguiente fórmula:


d₁(a, b) = |x₁ - x₂| + |y₁ - y₂|


Esta distancia se escribe como d₁(a, b), donde el subíndice 1 indica que elevamos las diferencias a la primera potencia 
(a diferencia de la distancia euclidiana, donde usamos potencias cuadradas).
Ahora, para ilustrar esta métrica con un ejemplo práctico, volvamos a nuestro mapa de Manhattan. 
Supón que una persona está en una dirección y queremos encontrar cuál taxi es el más cercano a ella, 
usando distancia Manhattan.

Aquí tienes el código que nos permite calcularlo:

"""
#EJERCICIO 1
import numpy as np


def manhattan_distance(first, second):
    return np.abs(first - second).sum()# < escribe tu código aquí >

    
first = np.array([3, 11])
second = np.array([1, 6])

print(manhattan_distance(first, second))

#EJERCICIO 2
'''
En una zona dada de Manhattan, vamos a encontrar el taxi más cercano de los tres que están desocupados.

Variables declaradas:

avenues_df— lista de avenidas con coordenadas
streets_df — lista de calles con coordenadas
address — ubicación del cliente (avenida y calle)
taxis — ubicación de los taxis

Fíjate en que, al igual que en el caso de la distancia euclidiana, 
podemos calcular las distancias desde la dirección hasta los coches utilizando la función correspondiente de la librería SciPy.
Si no puedes adivinar cómo se llama, echa un vistazo a la documentación 
del módulo de distancias de SciPy (materiales en inglés) o consulta la pista.

Calcula las distancias y guarda el resultado en la variable taxis_distance. 
Determina el número de serie del taxi más cercano. 
Muestra la ubicación del taxi (avenida y calle) en la pantalla (en precódigo).
'''
import numpy as np
import pandas as pd
from scipy.spatial import distance

# Crea DataFrames para las avenidas y calles
avenues_df = pd.DataFrame([0, 153, 307, 524], index=['Park', 'Lexington', '3rd', '2nd'], columns=['Distance'])
 
						  
streets_df = pd.DataFrame([0, 81, 159, 240, 324], index=['76', '75', '74', '73', '72'], columns=['Distance'])
 

# Datos de direcciones y taxis
address = ['Lexington', '74']
taxis = [
    ['Park', '72'],
    ['2nd', '75'],
    ['3rd', '76'],
]

# Convierte dirección en vector
address_vector = np.array([avenues_df.loc[address[0], 'Distance'], streets_df.loc[address[1], 'Distance']])

# Calcula distancias

taxi_distances = []
for taxi in taxis:
    taxis_vector =  np.array([avenues_df.loc[taxi[0], "Distance"], streets_df.loc[taxi[1], "Distance"]])
    dist = distance.cityblock(address_vector, taxis_vector)
    taxi_distances.append(dist)
# < escribe tu código aquí >

# Encuentra el índice del taxi más cercano
index = np.argmin(taxi_distances) # < escribe tu código aquí >
print(taxis[index])

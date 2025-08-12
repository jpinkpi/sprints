#Distancia euclidiana

"""
Distancia euclidiana

En esta lección aprenderemos a utilizar el producto escalar para encontrar la distancia entre los puntos del plano.
Empecemos por observar el vector a = (x, y).
¿Cómo podemos saber su longitud? Lo primero que podemos hacer es imaginar que nuestro vector es la hipotenusa de un triángulo rectángulo. 
Esto es útil porque, si recordamos un poco las propiedades de los triángulos rectángulos, podemos utilizar el teorema de Pitágoras para encontrar la longitud de nuestro vector.

Puede que te estés diciendo: "Muy bien, estupendo, pero ¿qué tiene eso que ver con el producto escalar?". Bueno, fíjate en lo que ocurre cuando calculas el producto escalar de un vector y de sí mismo:

a⋅a = (x*x)+(y*y) = x² + y²

Si sacamos la raíz cuadrada de esto, tendremos el mismo resultado que obtenemos gracias al teorema de Pitágoras:

√(x² + y²)

Está bastante bien, ¿verdad? Lo mejor de esto es que nos ofrece una forma práctica de calcular las longitudes de los vectores utilizando np.dot():
"""

import numpy as np

a = np.array([5, 6])

# mediante el teorema de Pitágoras
print(((a[0] ** 2) + (a[1] ** 2)) ** 0.5)

print()

# mediante el producto punto; ¿no es esto más bonito?
print(np.dot(a, a) ** 0.5)

"""
Ahora podemos utilizar esta información para encontrar la diferencia entre dos vectores: el vector a y el vector b.

Como puedes ver, podemos pensar en la diferencia entre los dos vectores como en un nuevo vector (b−a). Entonces, con el truco de antes, podemos calcular la longitud de este vector:

d₂(a, b) = (b-a)⋅(b-a) = (x₂-x₁)² + (y₂-y₁)²

Esta manera de medir la distancia se llama distancia euclidiana, que se utiliza en muchos algoritmos de machine learning como métrica de distancia por defecto. La distancia euclidiana es siempre la distancia más corta entre dos puntos en un plano.

Esta distancia se escribe como d₂(a, b), donde d tiene el subíndice 2 para indicar que las coordenadas del vector están elevadas a la segunda potencia.

Ahora vamos a utilizar NumPy para encontrar la distancia euclidiana entre a = (5,6) y b = (1,3):
"""

a = np.array([5, 6])
b = np.array([1, 3])
d = np.dot(b - a, b - a) ** 0.5
print('La distancia entre a y b es de', d)

"""
SciPy tiene una librería dedicada al cálculo de distancias, que se llama distance. Podemos importarla desde scipy.spatial y llamar a distance.euclidean() para calcular la distancia euclidiana:
"""

from scipy.spatial import distance

a = np.array([5, 6])
b = np.array([1, 3])
d = distance.euclidean(a, b)
print('La distancia entre a y b es de', d)

"""
Los resultados de los cálculos realizados mediante la fórmula y mediante la función son los mismos. La función puede utilizarse para calcular distancias tanto entre puntos como entre vectores que unen el origen y esos puntos. Calcularemos la distancia entre vectores en las tareas que vienen a continuación.

Aquí tienes un ejemplo. El mapa muestra las localidades que están ubicadas en el área de entrega de la empresa de entrega con drones Flying Stuff. Consideremos que Willowford es el origen. Las coordenadas del resto de las localidades están trazadas a lo largo de los ejes X e Y en kilómetros.

Para planificar la ruta del dron, almacenaremos los datos de las entregas en tres variables:

x_axis — coordenadas de cada población a lo largo del eje X  
y_axis — coordenadas a lo largo del eje Y  
shipments — número medio de entregas semanales en cada ciudad

Combinaremos estos datos en un DataFrame y los mostraremos en la pantalla:
"""

x_axis = np.array(
    [
        0.0,
        0.18078584,
        9.32526599,
        17.09628721,
        4.69820241,
        11.57529305,
        11.31769349,
        14.63378951,
    ]
)

y_axis = np.array(
    [
        0.0,
        7.03050245,
        9.06193657,
        0.1718145,
        5.1383203,
        0.11069032,
        3.27703365,
        5.36870287,
    ]
)

deliveries = np.array([5, 7, 4, 3, 5, 2, 1, 1])

town = [
    'Willowford',
    'Otter Creek',
    'Springfield',
    'Arlingport',
    'Spadewood',
    'Goldvale',
    'Bison Flats',
    'Bison Hills',
]

import pandas as pd

data = pd.DataFrame(
    {
        'x_coordinates_km': x_axis,
        'y_coordinates_km': y_axis,
        'Deliveries': deliveries,
    },
    index=town,
)

print(data)

"""
                    x_coordinates_km    y_coordinates_km    Deliveries
Willowford            0.000000            0.000000            5
Otter Creek           0.180786            7.030502            7 
Springfield           9.325266            9.061937            4
Arlingport           17.096287            0.171815            3
Spadewood             4.698202            5.138320            5
Goldvale             11.575293            0.110690            2
Bison Flats          11.317693            3.277034            1
Bison Hills          14.633790            5.368703            1

Calcularemos las distancias entre los puntos utilizando las longitudes de los vectores. Con este fin, vamos a extraer las coordenadas de los puntos en la variable llamada vectors:
"""

vectors = data[['x_coordinates_km', 'y_coordinates_km']].values
print(vectors)

"""
[[ 0.          0.        ]
 [ 0.18078584  7.03050245]
 [ 9.32526599  9.06193657]
 [17.09628721  0.1718145 ]
 [ 4.69820241  5.1383203 ]
 [11.57529305  0.11069032]
 [11.31769349  3.27703365]
 [14.63378951  5.36870287]]
"""

#EJERCICIO 1
'''
Crea una tabla con las distancias entre las localidades y guárdala en la variable distances. 
Presenta los datos en forma de lista de listas. Cada fila debe representar la distancia entre cada población y todas las demás.
Añade los nombres de todas las localidades a la tabla y muestra los resultados (en precódigo).
'''

import numpy as np
import pandas as pd
from scipy.spatial import distance

x_axis = np.array([0., 0.18078584, 9.32526599, 17.09628721,
                      4.69820241, 11.57529305, 11.31769349, 14.63378951])

y_axis  = np.array([0.0, 7.03050245, 9.06193657, 0.1718145,
                      5.1383203, 0.11069032, 3.27703365, 5.36870287])

deliveries = np.array([5, 7, 4, 3, 5, 2, 1, 1])

town = [
    'Willowford',
    'Otter Creek',
    'Springfield',
    'Arlingport',
    'Spadewood',
    'Goldvale',
    'Bison Flats',
    'Bison Hills',
]

data = pd.DataFrame(
    {
        'x_coordinates_km': x_axis,
        'y_coordinates_km': y_axis,
        'Deliveries': deliveries,
    },
    index=town,
)

vectors = data[['x_coordinates_km', 'y_coordinates_km']].values

distances = []
for town_from in range(len(town)):
    row= []
    for town_to in range(len(town)):
        value= distance.euclidean(vectors[town_from], vectors[town_to])
        row.append(value)
    distances.append(row)
# < escribe tu código aquí >
    
distances_df = pd.DataFrame(distances, index=town, columns=town)
print(distances_df)

#EJERCICIO 2
'''
El número de entregas semanales en cada ciudad ya se conoce. Gracias a toda esta información, puedes seleccionar la mejor población para ubicar el almacén de Flying Stuff.
Calcula para cada localidad cuántos kilómetros recorrerá el dron en una semana a condición de que el almacén se encuentre en esa ciudad.
 Busca las distancias entre poblaciones, duplícalas (ida y vuelta) y multiplícalas por el número de entregas semanales. Guarda el resultado en la lista  deliveries_in_week.
Encuentra la población con el menor recorrido total.
Muestra los resultados (en precódigo).
'''

import numpy as np
import pandas as pd
from scipy.spatial import distance

x_axis = np.array([0., 0.18078584, 9.32526599, 17.09628721,
                      4.69820241, 11.57529305, 11.31769349, 14.63378951])

y_axis  = np.array([0.0, 7.03050245, 9.06193657, 0.1718145,
                      5.1383203, 0.11069032, 3.27703365, 5.36870287])

deliveries = np.array([5, 7, 4, 3, 5, 2, 1, 1])

town = [
    'Willowford',
    'Otter Creek',
    'Springfield',
    'Arlingport',
    'Spadewood',
    'Goldvale',
    'Bison Flats',
    'Bison Hills',
]

data = pd.DataFrame(
    {
        'x_coordinates_km': x_axis,
        'y_coordinates_km': y_axis,
        'Deliveries': deliveries,
    },
    index=town,
)

vectors = data[['x_coordinates_km', 'y_coordinates_km']].values

distances = []
for town_from in range(len(town)):
    row = []
    for town_to in range(len(town)):
        value = distance.euclidean(vectors[town_from], vectors[town_to])
        row.append(value)
    distances.append(row)

distances = np.array(distances)
deliveries_in_week =[]
for i in range(len(town)):
    value= 2 * np.dot(np.array(distances[i]), deliveries)
    deliveries_in_week.append(value)
# < escribe tu código aquí >

deliveries_in_week_df = pd.DataFrame(
    {'Distance': deliveries_in_week}, index=town
)

print(deliveries_in_week_df)

print()
print('Localidad del almacén:', deliveries_in_week_df['Distance'].idxmin())
     # < escribe tu código aquí >)
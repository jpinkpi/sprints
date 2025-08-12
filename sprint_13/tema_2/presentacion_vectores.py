#Presentación de vectores

'''
Vamos a visualizar los vectores para examinar la lista de números en el plano de coordenadas.
Tracemos un vector bidimensional. Consta de dos números. El primero es la coordenada en el eje horizontal x y el segundo es la coordenada en el eje vertical y. 
El vector se representa mediante un punto o una flecha, que une el origen y el punto con coordenadas (x, y).

La razón de utilizar una flecha radica en que esta indica las dos componentes de un vector: magnitud y dirección. 
Por ejemplo, el vector [2, 3] es un desplazamiento de dos casillas hacia la derecha a lo largo del eje x y 
de tres casillas hacia arriba a lo largo del eje y. 
Si trabajamos con varios vectores que se encuentran en la misma línea, es mejor utilizar un punto para representar un vector.

Por ejemplo, vamos a trazar los vectores vector1 - [2, 3] y vector2 - [6, 2] utilizando puntos. Los puntos están dibujados con el método plt.plot().
'''

import numpy as np
import matplotlib.pyplot as plt

vector1 = np.array([2, 3])
vector2 = np.array([6, 2])

plt.figure(figsize=(10, 10))
plt.axis([0, 7, 0, 7])
# El argumento 'ro' establece el estilo del gráfico
# 'r' - rojo
# 'o' - círculo
plt.plot([vector1[0], vector2[0]], [vector1[1], vector2[1]], 'ro')
plt.grid(True)
plt.show()

'''
Vamos a utilizar flechas para dibujar los mismos vectores. En lugar de plt.plot(), llama a plt.arrow().
'''

import numpy as np
import matplotlib.pyplot as plt

vector1 = np.array([2, 3])
vector2 = np.array([6, 2])

plt.figure(figsize=(10, 10))
plt.axis([0, 7, 0, 7])
plt.arrow(
    0,
    0,
    vector1[0],
    vector1[1],
    head_width=0.3,
    length_includes_head="True",
    color='b',
)
plt.arrow(
    0,
    0,
    vector2[0],
    vector2[1],
    head_width=0.3,
    length_includes_head="True",
    color='g',
)
plt.plot(0, 0, 'ro')
plt.grid(True)
plt.show()


#EJERCICIO 1
'Traza el vector [75, 15]del ejemplo de LuxForVIP utilizando una flecha.'

import numpy as np
import matplotlib.pyplot as plt

vector = np.array([75, 15])
plt.figure(figsize=(3.5,3.5))
plt.axis([0, 100, 0, 100]) 
plt.arrow( 0, 0,vector[0], vector[1], head_width=4, length_includes_head="True")# < escribe el código aquí >, head_width=4, length_includes_head="True")
plt.xlabel('Price')
plt.ylabel('Quality')
plt.grid(True)
plt.show()

#EJERCICIO 2

"Traza todos los vectores bidimensionales con las valoraciones de los visitantes de LuxForVIP marcando puntos en el plano."

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ratings_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])

plt.figure(figsize=(3.5,3.5))
plt.axis([0, 100, 0, 100])
price = ratings['Price'].values
quality = ratings['Quality'].values
plt.plot(price,quality , "ro")# <  escribir código aquí  >, 'ro')
plt.xlabel('Price')
plt.ylabel('Quality')
plt.grid(True)
plt.show()

#EJERCICIO 3
'''
3.

Basado en el gráfico de la tarea anterior, crea dos listas separadas de vectores bidimensionales que contengan las valoraciones de los visitantes: 
para los visitantes que vinieron del agregador del mercado  de masas y otra para los que vinieron del agregador de marcas de lujo. 
Nombra las variables visitors_1 y visitors_2. 
Para ello, deberás especificar los umbrales de precio y calidad que dividen a los visitantes de ambos agregadores;
y luego aplicar estos umbrales para separarlos en las variables antes mencionadas.
Muestra sus valores en la pantalla (en precódigo).
'''


import numpy as np
import pandas as pd

ratings_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])

visitors_1 = []
visitors_2 = []
for visitor in list(ratings.values):
    price, quality = visitor
    if price < 40 and quality > 60:
        visitors_1.append([price, quality])
    else:
        visitors_2.append([price, quality])
    # < escribe tu código aquí  >

print('Valoraciones de los visitantes del primer agregador:', visitors_1)
print('Valoraciones de los visitantes del primer agregador:', visitors_2)
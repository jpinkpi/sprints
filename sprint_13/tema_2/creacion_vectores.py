#Creación de vectores
'''
Cualquier dato numérico puede representarse como un vector.
En Python, los conjuntos de datos a menudo se representan como listas. 
En matemáticas, un conjunto ordenado de datos numéricos es un vector, o un vector aritmético. Las operaciones que se puedan realizar con números, es decir, sumas, restas y multiplicaciones, también se podrán realizar con vectores. En Python, las operaciones con vectores son cientos de veces más rápidas que las operaciones con listas.

Pasemos a la librería NumPy. 
Su nombre se debe a las siglas de Numeric Python (Python Numérico). 
La librería proporciona herramientas para trabajar con vectores. 
Ya vimos np.arange(), que es una de sus funciones. 
En este curso, nos hará falta la función np.array(). No te dejes confundir por el nombre. 
Una matriz (array) es una forma útil de representar vectores en nuestro código, 
y a menudo usamos los términos indistintamente.

Crea una lista de dos números:
'''

numbers1 = [2, 3] # Lista de Python
print(numbers1)

#Convierte la lista en un vector:
import numpy as np

vector1 = np.array(numbers1) # Matriz de NumPy
print(vector1)

#Crea otro vector sin una variable temporal:
import numpy as np

vector2 = np.array([6, 2])
print(vector2)

#Convierte este vector en una lista:
numbers2 = list(vector2) # Lista a partir de vector
print(numbers2)

#La columna de la estructura de DataFrame en pandas
#se convierte en un vector NumPy utilizando el atributo values:

import pandas as pd

data = pd.DataFrame([1, 7, 3])
print(data[0].values)

#Utiliza la función len() para determinar el tamaño del vector (número de sus elementos):
print(len(vector2))

'''
Veamos un ejemplo concreto. La gente va a la tienda online de ropa LuxForVIP desde dos sitios agregadores. 
El primero anuncia artículos del mercado de masas, mientras que el segundo sitio agrega ropa de marca de lujo. 
Los visitantes de LuxForVIP valoran su satisfacción con los precios y la calidad de los productos en una escala de 0 a 100.
La tabla contiene las valoraciones de todos los visitantes. 
Una puntuación alta para el precio indica que al cliente le gustó el precio (es decir, que la relación calidad-precio del artículo es buena). 
Del mismo modo, una puntuación alta para la calidad indica que el cliente consideró que el artículo era de alta calidad.
'''

import numpy as np
import pandas as pd

ratings_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])
print(ratings)


#EJERCICIO 1
'''
1.

Crea dos vectores. El primero contiene todas las valoraciones del precio. 
El segundo contiene las valoraciones de la calidad.
'''
import numpy as np
import pandas as pd

ratings_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])

price = np.array(ratings["Price"]) # < escribe tu código aquí >
quality = np.array(ratings["Quality"]) # < escribe tu código aquí >
print('Precio: ', price)
print('Calidad: ', quality)


#EJERCICIO 2
'''
Encuentra el número total de visitantes de LuxForVIP. Encuentra el vector con las valoraciones del visitante 4 (quinta fila del dataframe)
'''

ratings_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])

visitors_count = len(ratings) # < escribe tu código aquí >
visitor4 = np.array(ratings.iloc[4]) # < escribe tu código aquí >
print('Número de visitantes:', visitors_count)
print('Visitante 4:', visitor4)

#EJERCICIO 3

'''
La tabla tiene el atributo values.  values es una matriz bidimensional. 
Llama a la función list() para convertir esta matriz en una lista de vectores con las valoraciones de todos los visitantes.
'''


ratings_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])

vector_list = list(ratings.values)# < escribe tu código aquí >
print(vector_list)
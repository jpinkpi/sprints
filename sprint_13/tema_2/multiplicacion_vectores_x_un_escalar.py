#Multiplicación de un vector por un escalar
"""
Además de sumar y restar los vectores, también los podemos multiplicar por escalares.
Cada coordenada del vector se multiplica por el mismo número.

Vector        Coordenadas
x⃗            (x1, x2, ..., xn)
k·x⃗          (k·x1, k·x2, ..., k·xn)

En caso de que el número sea negativo, todas las coordenadas también cambiarán de signo.

Ejemplo:
Multiplicamos el vector1 — [2, 3] por un número positivo (k = 2).
"""

import numpy as np

vector1 = np.array([2, 3])
vector3 = 2 * vector1

print("Resultado de la multiplicación escalar:", vector3)

"Ahora lo multiplicaremos por un número negativo:"
vector4 = -1 * vector1
print(vector4)

'Cuando se trata de la multiplicación por un número positivo, los vectores mantienen su dirección en el plano, aunque las flechas cambian de longitud. Cuando se trata de la multiplicación por un número negativo, los vectores cambian al sentido opuesto, además de ser escalados'

import numpy as np
import matplotlib.pyplot as plt

vector1 = np.array([2, 3])
vector3 = 2 * vector1
vector4 = -1 * vector1

plt.figure(figsize=(10.2, 10))
plt.axis([-4, 6, -4, 6])
arrow1 = plt.arrow(
    0,
    0,
    vector1[0],
    vector1[1],
    head_width=0.3,
    length_includes_head="True",
    color='b',
)
arrow3 = plt.arrow(
    0,
    0,
    vector3[0],
    vector3[1],
    head_width=0.3,
    length_includes_head="True",
    color='g',
)
arrow4 = plt.arrow(
    0,
    0,
    vector4[0],
    vector4[1],
    head_width=0.3,
    length_includes_head="True",
    color='m',
)
plt.plot(0, 0, 'ro')
plt.legend(
    [arrow1, arrow3, arrow4],
    ['vector1', 'vector3', 'vector4'],
    loc='upper left',
)
plt.grid(True)
plt.show()

'Aquí tenemos un ejemplo en el que añadimos los precios a los datos de la tienda online después de la fusión.'

import numpy as np
import pandas as pd

quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = [
    'Silicone case for iPhone 8',
    'Leather case for iPhone 8',
    'Silicone case for iPhone XS',
    'Leather case for iPhone XS',
    'Silicone case for iPhone XS Max',
    'Leather case for iPhone XS Max',
    'Silicone case for iPhone 11',
    'Leather case for iPhone 11',
    'Silicone case for iPhone 11 Pro',
    'Leather case for iPhone 11 Pro',
]
stocks_1 = pd.DataFrame({'Quantity': quantity_1}, index=models)
quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Quantity': quantity_2}, index=models)

vector_of_quantity_1 = stocks_1['Quantity'].values
vector_of_quantity_2 = stocks_2['Quantity'].values
vector_of_quantity_united = vector_of_quantity_1 + vector_of_quantity_2

stocks_united = pd.DataFrame(
    {'Quantity': vector_of_quantity_united}, index=models
)
stocks_united['Price'] = [30, 21, 32, 22, 18, 17, 38, 12, 23, 29]
print(stocks_united)

#EJERCICIO 1
"""
Toma la columna de precios y conviértela en un vector numérico. Muestra los resultados (en precódigo).
"""

import numpy as np
import pandas as pd

quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = [
    'Silicone case for iPhone 8',
    'Leather case for iPhone 8',
    'Silicone case for iPhone XS',
    'Leather case for iPhone XS',
    'Silicone case for iPhone XS Max',
    'Leather case for iPhone XS Max',
    'Silicone case for iPhone 11',
    'Leather case for iPhone 11',
    'Silicone case for iPhone 11 Pro',
    'Leather case for iPhone 11 Pro',
]
stocks_1 = pd.DataFrame({'Quantity' : quantity_1}, index=models)
quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Quantity' : quantity_2}, index=models)

vector_of_quantity_1 = stocks_1['Quantity'].values
vector_of_quantity_2 = stocks_2['Quantity'].values
vector_of_quantity_united = vector_of_quantity_1 + vector_of_quantity_2

stocks_united = pd.DataFrame(
    {'Quantity': vector_of_quantity_united}, index=models
)
stocks_united['Price'] = [30, 21, 32, 22, 18, 17, 38, 12, 23, 29]

price_united = stocks_united['Price'].values# < escribe tu código aquí >
print(price_united)



#EJERCICIO 2
'''
FoCaseology ha anunciado un descuento del 10 % en toda su gama. Encuentra el nuevo vector de precios, teniendo en cuenta el descuento. 
Muestra los resultados (en precódigo).
'''

import numpy as np
import pandas as pd

quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = [
    'Silicone case for iPhone 8',
    'Leather case for iPhone 8',
    'Silicone case for iPhone XS',
    'Leather case for iPhone XS',
    'Silicone case for iPhone XS Max',
    'Leather case for iPhone XS Max',
    'Silicone case for iPhone 11',
    'Leather case for iPhone 11',
    'Silicone case for iPhone 11 Pro',
    'Leather case for iPhone 11 Pro',
]
stocks_1 = pd.DataFrame({'Quantity': quantity_1}, index=models)
quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Quantity': quantity_2}, index=models)

vector_of_quantity_1 = stocks_1['Quantity'].values
vector_of_quantity_2 = stocks_2['Quantity'].values
vector_of_quantity_united = vector_of_quantity_1 + vector_of_quantity_2

stocks_united = pd.DataFrame(
    {'Quantity': vector_of_quantity_united}, index=models
)
stocks_united['Price'] = [30, 21, 32, 22, 18, 17, 38, 12, 23, 29]

price_united = stocks_united['Price'].values

price_discount_10 = -price_united * -.90 # < escribe tu código aquí >
stocks_united['10% discount price'] = price_discount_10.astype(int)
print(stocks_united)


#EJERCICIO 3
'''
Cuando las rebajas terminaron, FoCaseology subió los precios un 10 %. 
Crea la lista de los precios aumentados. Muestra los resultados (en precódigo).
'''
import numpy as np
import pandas as pd

quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = [
    'Silicone case for iPhone 8',
    'Leather case for iPhone 8',
    'Silicone case for iPhone XS',
    'Leather case for iPhone XS',
    'Silicone case for iPhone XS Max',
    'Leather case for iPhone XS Max',
    'Silicone case for iPhone 11',
    'Leather case for iPhone 11',
    'Silicone case for iPhone 11 Pro',
    'Leather case for iPhone 11 Pro',
]
stocks_1 = pd.DataFrame({'Quantity': quantity_1}, index=models)
quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Quantity': quantity_2}, index=models)

vector_of_quantity_1 = stocks_1['Quantity'].values
vector_of_quantity_2 = stocks_2['Quantity'].values
vector_of_quantity_united = vector_of_quantity_1 + vector_of_quantity_2

stocks_united = pd.DataFrame(
    {'Quantity': vector_of_quantity_united}, index=models
)
stocks_united['Price'] = [30, 21, 32, 22, 18, 17, 38, 12, 23, 29]

price_united = stocks_united['Price'].values
price_discount_10 = price_united * 0.9
stocks_united['10% discount prise'] = price_discount_10.astype(int)

price_no_discount = price_discount_10 * 1.10 # < escribe tu código aquí >
stocks_united['10% raise price'] = price_no_discount.astype(int)
print(stocks_united)

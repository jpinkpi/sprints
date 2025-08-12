import numpy as np
import pandas as pd

'''
MULTIPLICACIÓN DE MATRICES

En la multiplicación de matrices, se construye una nueva matriz a partir de dos matrices existentes.
Cada elemento del resultado es el producto escalar de una fila de la primera matriz y una columna de la segunda.

Regla clave:
→ Si A es de tamaño (m x n) y B es de tamaño (n x p),
→ entonces el resultado será una matriz C de tamaño (m x p).

La dimensión "n" debe coincidir entre las matrices para que la multiplicación sea válida.
'''

# Matrices A y B
A = np.array([
    [1, 2, 3], 
    [-1, -2, -3]
])

B = np.array([
    [1, 0], 
    [0, 1],
    [1, 1]
])

print("Multiplicación de matrices A @ B:")
print(np.dot(A, B))     # Forma 1
print(A.dot(B))         # Forma 2
print(A @ B)            # Forma 3 (más moderna)

'''
Todas las formas anteriores son equivalentes. 
Resultado:
[[ 4  5]
 [-4 -5]]

Nota: Cambiar el orden de los factores cambia el resultado.
'''

print("\nMultiplicación de B @ A:")
print(B @ A)
# Resultado:
# [[ 1  2  3]
#  [-1 -2 -3]
#  [ 0  0  0]]

'''
Si tratamos de multiplicar una matriz por sí misma sin que sea cuadrada, obtenemos un error.
'''

matrix = np.array([
    [1, 2, 3], 
    [-1, -2, -3]
])

print("\nIntento de multiplicar matriz no cuadrada consigo misma:")
try:
    print(matrix @ matrix)
except ValueError as e:
    print("Error:", e)

'''
Solo podemos multiplicar una matriz por sí misma si es cuadrada.
Ejemplo:
'''

square_matrix = np.array([
    [1, 2, 3], 
    [-1, -2, -3],
    [0, 0, 0]
])

print("\nMultiplicación de matriz cuadrada consigo misma:")
print(square_matrix @ square_matrix)

'''
EJEMPLO PRÁCTICO: HOTEL

El hotel tiene 2 tipos de habitaciones:
- Suite luna de miel: $700 por noche
- Habitación regular: $50 por noche

La matriz de precios se representa con una fila y dos columnas.
'''

costs = np.array([
    [700, 50]  # [suite, regular]
])

'''
Reservas de habitaciones por mes (2 periodos):
Periodo 1: febrero y marzo
Periodo 2: junio, julio y agosto
'''

bookings_1 = np.array([
    [3, 4],     # suites
    [17, 26]    # habitaciones regulares
])

bookings_2 = np.array([
    [8, 9, 11],     # suites
    [21, 26, 30]    # habitaciones regulares
])

months_1 = ['February', 'March']
months_2 = ['June', 'July', 'August']

'''
Multiplicamos precios x reservas para obtener ingresos mensuales.
'''

revenue_1 = costs @ bookings_1
revenue_2 = costs @ bookings_2

print("\nIngresos por mes (febrero-marzo):")
print(pd.DataFrame(revenue_1, columns=months_1))

print("\n----------------------")

print("Ingresos por mes (junio-agosto):")
print(pd.DataFrame(revenue_2, columns=months_2))

'''
Resultados:
   February  March
0      2950   4100
----------------------
   June  July  August
0  6650  7600    9200

Esto demuestra cómo la multiplicación de matrices se aplica a un problema del mundo real para calcular ingresos automáticamente.
'''

#Ejercicio 1
'''
T-phone está de vuelta con otra tarea. Tienes una matriz que contiene datos sobre los paquetes de contenido de los clientes durante un mes.

Calcula la cantidad de minutos, mensajes de texto y megabytes gastados por todos los clientes durante el mes.
Muestra en la pantalla el resultado en la matriz clients_services con filas correspondientes a clientes y columnas correspondientes a servicios.
'''

import numpy as np
import pandas as pd

services = ['Minutos', 'Mensajes', 'Megabytes']
packs_names = ['"Detrás del volante', '"En el metro"']
packs = np.array([
    [20, 5],
    [2, 5],
    [500, 1000]])

clients_packs = np.array([
    [1, 2],
    [2, 3],
    [4, 1],
    [2, 3],
    [5, 0]])

print('Paquetes')
print(pd.DataFrame(clients_packs, columns=packs_names))
print()

clients_services = clients_packs @ packs.T 

print('Minutos, Mensajes y Megabytes')
print(pd.DataFrame(clients_services, columns=services))
print()

#Ejercicio 2
'''

Roble y Pino ha recibido pedidos de muebles de varios lugares diferentes y necesitan saber las cantidades de los materiales que necesitan para cumplir con los pedidos. 
Como antes, la matriz manufacture muestra los materiales necesarios para cada uno de los diferentes tipos de muebles, donde las filas son los tipos de muebles (silla, banca y mesa)
y las columnas son los materiales. Por ejemplo, para hacer una silla, necesitamos 0.2 metros de tabla, 1.2 metros de tubo de metal y 8 tornillos.

La matriz furniture muestra el número de cada mueble en los diferentes lugares: una cafetería, un comedor y un restaurante, 
donde las filas son los lugares y las columnas son los tipos de muebles. Por ejemplo, la cafetería necesita 12 sillas, 0 bancas y 3 mesas.

Encuentra la matriz venues_materials, que contiene la cantidad de materiales (columnas) para cada establecimiento (filas). 
Muestra el resultado en pantalla (en precódigo).

Calcula la cantidad de materiales necesarios para la producción de muebles si el pedido se recibió de 18 cafeterías, 12 comedores y 7 restaurantes. 
Guarda este vector en la variable total_materials y muéstralo en la pantalla (en precódigo).
'''

import numpy  as np
import pandas as pd

materials_names = ['Tabla', 'Tubos', 'Tornillos']
venues_names = ['Cafetería', 'Comedor', 'Restaurante']

manufacture = np.array([
    [0.2, 1.2, 8],    # materiales para silla
    [0.5, 0.8, 6],    # materiales para banco
    [0.8, 1.6, 8]])   # materiales para mesa

furniture = np.array([
    [12, 0, 3],   # Pedido de cafetería (silla, banca, mesa)
    [40, 2, 10],  # Pedido de comedor
    [60, 6, 18]]) # Pedido de restaurante

venues_materials = furniture @ manufacture # < escribe tu código aquí >

print('Por el establecimiento')
print(pd.DataFrame(venues_materials, index=venues_names, columns=materials_names))
print()

venues = [18, 12, 7]

total_materials = venues @ venues_materials  # < escribe tu código aquí >

print('Total')
print(pd.DataFrame([total_materials], index=[''], columns=materials_names))
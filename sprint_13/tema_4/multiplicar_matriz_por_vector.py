#Multiplicar una matriz por un vector
import numpy as np
import pandas as pd

'''
MULTIPLICAR UNA MATRIZ POR UN VECTOR

¿Qué pasa si multiplicas una matriz por un vector? ¡Obtienes un vector!

Al principio puede parecer una tarea imposible; después de todo, con tantos números, ¿por dónde empezar?
Pero, como veremos, el proceso no solo es lógico, sino también extremadamente útil al agregar datos o construir modelos de machine learning.

Para entender exactamente cómo se multiplica una matriz por un vector, primero debemos recordar que una matriz no es más que un conjunto de vectores de la misma longitud.

Ya conocemos el uso del producto escalar para multiplicar un vector por otro.
El mismo principio guía este proceso también: para multiplicar una matriz por un vector, tomamos el producto escalar de cada fila y el vector.

El resultado es un nuevo vector.

IMPORTANTE: la longitud de cada fila de la matriz debe ser igual a la longitud del vector.
'''

# Ejemplo básico con np.dot()
matrix = np.array([
    [1, 2, 3], 
    [4, 5, 6]])

vector = np.array([7, 8, 9])

print("Multiplicación de matriz por vector con np.dot():")
print(np.dot(matrix, vector))     # Función independiente
print(matrix.dot(vector))         # Método de la matriz
# Resultado:
# [ 50 122]

'''
En este ejemplo tenemos:
Fila 1: 1*7 + 2*8 + 3*9 = 50
Fila 2: 4*7 + 5*8 + 6*9 = 122

Podemos realizar esta operación porque el vector tiene tamaño 3 y la matriz tiene 3 columnas.
'''

'''
EJEMPLO PRÁCTICO – PAQUETES DE TELÉFONO (T-Phone)

Un operador móvil ofrece dos paquetes de servicios: "Al volante" y "En el metro".
Cada uno incluye una cantidad fija de minutos, mensajes y megabytes.

Los clientes compran diferentes cantidades de estos paquetes cada mes.
Queremos saber el consumo total de cada cliente.
'''

# Definimos los servicios y los nombres de paquetes
services = ['Minutos', 'Mensajes', 'Megabytes']
packs_names = ['"Al volante"', '"En el metro"']

# Contenido de cada paquete: filas = servicios, columnas = paquetes
packs = np.array([
    [20, 5],     # Minutos
    [2, 5],      # Mensajes
    [500, 1000]  # Megabytes
])

print("\nContenido de cada paquete (por unidad):")
print(pd.DataFrame(packs, columns=packs_names, index=services))

# Paquetes comprados por los clientes
clients_packs = np.array([
    [1, 2],  # Cliente 1
    [2, 3],  # Cliente 2
    [4, 1],  # Cliente 3
    [2, 3],  # Cliente 4
    [5, 0]   # Cliente 5
])

print("\nCantidad de paquetes comprados por cliente:")
print(pd.DataFrame(clients_packs, columns=packs_names))

'''
Ahora multiplicamos la matriz "packs" (servicios por paquete) por la matriz "clients_packs.T"
para obtener el consumo total de cada cliente.
'''

# Multiplicación: cada columna del resultado es el consumo total de un cliente
total_usage = np.dot(packs, clients_packs.T)

print("\nConsumo total por cliente (Minutos, Mensajes, Megabytes):")
print(pd.DataFrame(total_usage.T, columns=services))

'''
Explicación:
- packs es una matriz 3x2 (servicios x paquetes)
- clients_packs.T es una matriz 2x5 (paquetes x clientes)
- El resultado es una matriz 3x5 (servicios x clientes), que transponemos para obtener una forma amigable.

Así, para cada cliente obtenemos el número total de minutos, mensajes y megas que tiene disponible al mes.
'''

#Ejercicio 1
'''
Calcula la cantidad de minutos, mensajes de texto y 
tráfico de Internet gastados por el segundo cliente en el transcurso del mes. 
Guarda el resultado en la variable client_services y muéstralo en la pantalla (ya en el precódigo).
'''

import numpy as np
import pandas as pd

services = ['Minutos', 'Mensajes', 'Megabytes']
packs_names = ['"Al volante"', '"En el metro"']
packs = np.array([[20, 5], [2, 5], [500, 1000]])

clients_packs = np.array([[1, 2], [2, 3], [4, 1], [2, 3], [5, 0]])

print("Paquete del cliente")
print(pd.DataFrame(clients_packs[1], index=packs_names, columns=['']))
print()

client_vector = clients_packs[1,:] # < escribe tu código aquí >
client_services = np.dot(packs,client_vector)# < escribe tu código aquí >

print('Minutos', 'Mensajes', 'Megabytes')
print(client_services)
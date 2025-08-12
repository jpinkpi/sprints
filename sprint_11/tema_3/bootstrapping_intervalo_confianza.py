#Bootstrapping para el intervalo de confianza
'''
El Banco de Salvación Monetaria está probando un nuevo sistema de atención al cliente cuyo objetivo es determinar 
cuánto tiempo pasan los clientes en las colas de espera. Después de las pruebas, disponemos de una muestra de valores de tiempo de espera.
'''
import pandas as pd

data = pd.Series([
    10.7,  9.58,  7.74,  8.3 , 11.82,  9.74, 10.18,  8.43,  8.71,
     6.84,  9.26, 11.61, 11.08,  8.94,  8.44, 10.41,  9.36, 10.85,
    10.41,  8.37,  8.99, 10.17,  7.78, 10.79, 10.61, 10.87,  7.43,
     8.44,  9.44,  8.26,  7.98, 11.27, 11.61,  9.84, 12.47,  7.8,
    10.54,  8.99,  7.33,  8.55,  8.06, 10.62, 10.41,  9.29,  9.98,
     9.46,  9.99,  8.62, 11.34, 11.21, 15.19, 20.85, 19.15, 19.01,
    15.24, 16.66, 17.62, 18.22, 17.2, 15.76, 16.89, 15.22, 18.7,
    14.84, 14.88, 19.41, 18.54, 17.85, 18.31, 13.68, 18.46, 13.99,
    16.38, 16.88, 17.82, 15.17, 15.16, 18.15, 15.08, 15.91, 16.82,
    16.85, 18.04, 17.51, 18.44, 15.33, 16.07, 17.22, 15.9, 18.03,
    17.26, 17.6, 16.77, 17.45, 13.73, 14.95, 15.57, 19.19, 14.39,
    15.76])

'''
La métrica objetivo del experimento es el percentil del 99% de esta distribución, 
que se centra en los tiempos de espera más largos experimentados por la gran mayoría de los clientes (99%), 
excluyendo sólo los casos más extremos. Además, al banco no solo le interesa cuál es el tiempo de espera del percentil 99, 
sino también qué tan seguros están de esa estimación.

La administración del banco necesita implementar características confiables, 
por lo que se debe determinar el intervalo de confianza del 90% para el percentil del 99%.

¿Por qué podría ser útil esta estrategia para el banco?
Esta estrategia de bootstrapping es útil porque permite al banco estimar un rango de valores plausibles para el percentil 99 de los tiempos de espera de los clientes, 
en lugar de solo una estimación puntual. Esto es crucial para la toma de decisiones porque reconoce la incertidumbre inherente al trabajar con una muestra de datos.

El Bootstrapping proporciona una forma de cuantificar esta incertidumbre al calcular un intervalo de confianza.  
El intervalo de confianza del 90% indica que si se repitiera este proceso de prueba muchas veces, 
el 90% de los intervalos calculados contendrían el verdadero tiempo de espera del percentil 99.

En primer lugar, vamos a averiguar cómo formar submuestras para el bootstrapping. Ya conoces la función sample(). 
Para esta tarea necesitamos llamarla en un bucle. Pero aquí nos encontramos con un problema:
'''
for i in range(5):
    # extrae un elemento aleatorio de sample 1
    # especifica random_state para la reproducibilidad
    print(data.sample(1, random_state=12345))

"Como especificamos el random_state, el elemento aleatorio es siempre el mismo. Para solucionarlo, crea una instancia RandomState() del módulo  numpy.random:"
from numpy.random import RandomState

state = RandomState(12345)

for i in range(5):
    # extrae un elemento aleatorio de la muestra 1
    print(data.sample(1, random_state=state))

'''
Otro detalle importante a la hora de crear submuestras consiste en que deben proporcionar una selección de elementos con reemplazo. 
Es decir, el mismo elemento puede caer en una submuestra varias veces. 
Para ello, especifica replace=True para la función sample(). Compara:
'''
example_data = pd.Series([1, 2, 3, 4, 5])
print('Sin reemplazo')
print(example_data.sample(frac=1, replace=False, random_state=state))
print('Con reemplazo')
print(example_data.sample(frac=1, replace=True, random_state=state))

#EJERCICIO 1
'''
Crea 10 submuestras de bootstrap. Para cada submuestra, calcula el cuantil del 99% después de realizar un muestreo
aleatorio de 15 puntos de datos (con reemplazo). Muestra en pantalla cada punto percentil.

Comprueba la función quantile() para las instancias de pandas.Series
'''

import pandas as pd
import numpy as np

data = pd.Series([
    10.7 ,  9.58,  7.74,  8.3 , 11.82,  9.74, 10.18,  8.43,  8.71,
     6.84,  9.26, 11.61, 11.08,  8.94,  8.44, 10.41,  9.36, 10.85,
    10.41,  8.37,  8.99, 10.17,  7.78, 10.79, 10.61, 10.87,  7.43,
     8.44,  9.44,  8.26,  7.98, 11.27, 11.61,  9.84, 12.47,  7.8 ,
    10.54,  8.99,  7.33,  8.55,  8.06, 10.62, 10.41,  9.29,  9.98,
     9.46,  9.99,  8.62, 11.34, 11.21, 15.19, 20.85, 19.15, 19.01,
    15.24, 16.66, 17.62, 18.22, 17.2 , 15.76, 16.89, 15.22, 18.7 ,
    14.84, 14.88, 19.41, 18.54, 17.85, 18.31, 13.68, 18.46, 13.99,
    16.38, 16.88, 17.82, 15.17, 15.16, 18.15, 15.08, 15.91, 16.82,
    16.85, 18.04, 17.51, 18.44, 15.33, 16.07, 17.22, 15.9 , 18.03,
    17.26, 17.6 , 16.77, 17.45, 13.73, 14.95, 15.57, 19.19, 14.39,
    15.76])

state = np.random.RandomState(12345)

for i in range(10):
    subsample = data.sample(frac=.15,replace=True, random_state=state) # < escribe tu código aquí >
    print(subsample.quantile(q=.99))# < escribe tu código aquí >))


#EJERCICIO 2
'''

Esta vez, tu tarea consistirá en estimar un intervalo de confianza del 90% para el percentil 99 de los tiempos de espera de los clientes utilizando la técnica del bootstrapping. 
Esto es importante porque nos proporciona un rango de valores plausibles sobre cuánto tiempo podrían tener que esperar los clientes, en lugar de una única suposición. 
Para ello, utilizaremos los datos (data) proporcionados (la muestra de tiempos de espera).

Modifica tu código de la tarea 1 guardándolo en una lista, cada valor del percentil 99 se obtiene de cada submuestra. 
Luego, obtén el quantile() inferior (lower) y superior (upper) para la lista del percentil 99 de 10 submuestras. 
Aquí obtendrás los percentiles 5 y 95 para el intervalo de confianza.
'''
import pandas as pd
import numpy as np

data = pd.Series([
    10.7 ,  9.58,  7.74,  8.3 , 11.82,  9.74, 10.18,  8.43,  8.71,
     6.84,  9.26, 11.61, 11.08,  8.94,  8.44, 10.41,  9.36, 10.85,
    10.41,  8.37,  8.99, 10.17,  7.78, 10.79, 10.61, 10.87,  7.43,
     8.44,  9.44,  8.26,  7.98, 11.27, 11.61,  9.84, 12.47,  7.8 ,
    10.54,  8.99,  7.33,  8.55,  8.06, 10.62, 10.41,  9.29,  9.98,
     9.46,  9.99,  8.62, 11.34, 11.21, 15.19, 20.85, 19.15, 19.01,
    15.24, 16.66, 17.62, 18.22, 17.2 , 15.76, 16.89, 15.22, 18.7 ,
    14.84, 14.88, 19.41, 18.54, 17.85, 18.31, 13.68, 18.46, 13.99,
    16.38, 16.88, 17.82, 15.17, 15.16, 18.15, 15.08, 15.91, 16.82,
    16.85, 18.04, 17.51, 18.44, 15.33, 16.07, 17.22, 15.9 , 18.03,
    17.26, 17.6 , 16.77, 17.45, 13.73, 14.95, 15.57, 19.19, 14.39,
    15.76])

state = np.random.RandomState(12345)

# Guarda los valores del cuantil del 99 % en la variable de valores
values = []
for i in range(1000):
    subsample = data.sample(frac=0.15, replace=True, random_state=state)
    values.append(subsample.quantile(.99))# < escribe tu código aquí >
values = pd.Series(values)
# < escribe tu código aquí >
    
lower = values.quantile(q=.05)# < escribe tu código aquí >
upper = values.quantile(q=.95)# < escribe tu código aquí >

print(lower)



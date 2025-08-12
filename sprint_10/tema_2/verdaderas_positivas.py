#Respuestas verdaderas positivas
'''
Podemos utilizar diferentes métricas para gestionar el desequilibrio y clasificar las respuestas con mayor exactitud.
¿Qué significa una respuesta verdadera positiva (VP)? En este caso, el modelo etiquetó una observación como 1 y su valor real también es 1.
En nuestro ejercicio, la respuesta Verdadero Positivo equivale al número de asegurados que:
pidió compensación, de acuerdo con la predicción del modelo;
sí hizo un reclamo de seguro.
'''

#Ejercicio 

'''
Aquí hay un ejemplo de predicciones frente a respuestas correctas. Cuenta el número de respuestas VP y muestra el resultado en la pantalla.
'''

import pandas as pd

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

true_positives = ((target == 1 )& (predictions == 1)).sum()
print(true_positives)
# <  escribe el código aquí >



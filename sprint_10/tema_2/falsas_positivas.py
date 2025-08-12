#Respuestas falsas positivas
'''
Los algoritmos son como los humanos: cometen errores. Estos errores se dividen en dos categorías.
El primer tipo de error es un falso positivo (FP). Ocurre cuando el modelo predijo "1", pero el valor real de la clase es "0".

En nuestra tarea, una respuesta Falso Positivo es el número de personas aseguradas que:

según la predicción del modelo, solicitaron un pago, pero
en realidad, no hicieron un reclamo.
'''

#EJERCICIO
'''
Encuentra el número de respuestas FP de la misma manera que encontraste las respuestas VN en la tarea anterior. Muestra los resultados en la pantalla.
'''


import pandas as pd

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])
fake_positives = ((target == 0)& (predictions==1)).sum()
print(fake_positives)
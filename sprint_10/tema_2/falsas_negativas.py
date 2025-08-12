#Respuestas falsas negativas

'''
El segundo tipo de error son las respuestas Falsas Negativas (FN).
Las respuestas falsas negativas ocurren cuando el modelo predice "0", pero el valor real de la clase es "1".

En nuestra tarea, una respuesta de Falso Negativo es el número de personas aseguradas que:

según la predicción del modelo, no solicitaron un pago, pero
en realidad, hicieron un reclamo.
'''


#Ejercicio 
'''
Cuenta el número de respuestas FN. Muestra los resultados en la pantalla.
'''
import pandas as pd

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])
fake_negatives = ((target == 1) & (predictions== 0)).sum()
print(fake_negatives)
# < escribe el código aquí >


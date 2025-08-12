#Respuestas verdaderas negativas
'''
Si los valores de clase previstos y reales son negativos, la respuesta es Verdadero Negativo.
En nuestro ejercicio, la respuesta Verdadero Negativo (VN) es el número de personas aseguradas que:

de acuerdo con la predicción del modelo, no solicitaron un pago
en realidad no solicitaron la compensación del seguro.
'''

#EJERCICIO
"Cuenta el número de respuestas VN tal como lo hiciste en el ejercicio anterior. Muestra los resultados en la pantalla."

import pandas as pd

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])
true_negative = ((target == 0) & (predictions==0)).sum()
print(true_negative)


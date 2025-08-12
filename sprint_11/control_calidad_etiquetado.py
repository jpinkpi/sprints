#Control de calidad de etiquetado
'''
¿Qué pasa si el etiquetador comete un error? Afectará las predicciones del modelo. Averigüemos cómo evitar esto.

La calidad de los datos después del etiquetado se puede mejorar utilizando los métodos para el control de calidad del etiquetado. 
¿Cómo funcionan? Todas las observaciones, o una parte de ellas, se etiquetan varias veces y luego se forma la respuesta final.

Veamos uno de esos métodos, el voto mayoritario. ¿Quién está "votando" y cómo? Por ejemplo, cada observación está etiquetada por tres evaluadores. 
La respuesta final es la elegida por la mayoría.

Aquí verás cómo este método funciona con un conjunto de datos. Una empresa médica lanza un sistema para el diagnóstico automatizado de enfermedades del corazón. 
Los datos de 303 pacientes están etiquetados por tres profesionales de la salud.


Las siguientes características están disponibles para nosotros:

age — edad del paciente
sex — sexo biológico del paciente
cp — tipo de dolor en el pecho
trestbps — presión arterial en reposo
chol — colesterol sérico
fbs — azúcar en sangre en ayunas (si > 120 mg/dl)
restecg — resultados electrocardiográficos en reposo
thalach — frecuencia cardíaca máxima alcanzada
exang — angina inducida por el ejercicio
oldpeak — depresión del ST inducida por el ejercicio en relación con el reposo
slope — la pendiente del segmento ST del ejercicio máximo
ca — número de vasos principales (0-3) coloreados por fluoroscopia
thal — Resultado de la prueba de esfuerzo con talio
Etiquetado:

label_1 — respuesta del Profesional 1
label_2 — respuesta del Profesional 2
label_3 — respuesta del Profesional 3
'''


#Eercicio 1

'''
Realiza el voto mayoritario para el conjunto de datos. 
Almacena el objetivo en la variable objetivo. Muestra en pantalla las cinco primeras filas de la tabla resultante (en precódigo).

Hay muchas soluciones al problema, elige la que más te convenga.
'''

try:
    import pandas as pd

    data = pd.read_csv('/datasets/heart_labeled.csv')

    target = []
    for i in range(data.shape[0]):
        labels = data.loc[i, ["label_1", "label_2", "label_3"]]# < escribe tu código aquí >]
        true_label = int(labels.mean() >0.5) # < escribe tu código aquí >
        target.append(true_label)
    data['target'] = target


    # < escribe tu código aquí >

    print(data.head())
except:print("prueba")
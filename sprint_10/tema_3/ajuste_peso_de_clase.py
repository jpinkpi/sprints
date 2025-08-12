#Ajuste de peso de clase
'''

¡Hagamos que las clases raras pesen más!
Imagina que estás haciendo un examen. Dependiendo de la dificultad, las respuestas correctas pueden darte uno o dos puntos. 
Para obtener una puntuación más alta, te enfocas solo en las respuestas de dos puntos. 
De manera similar, los modelos también buscan una puntuación más alta, por lo que tienden a centrarse en las observaciones de mayor importancia.
Los algoritmos de aprendizaje automático consideran que todas las observaciones del conjunto de entrenamiento tienen la misma ponderación de forma predeterminada. 
Si necesitamos indicar que algunas observaciones son más importantes, asignamos un peso a la clase respectiva.


El algoritmo de regresión logística en la librería sklearn tiene el argumento class_weight. 
Por defecto, es None, es decir, las clases son equivalentes:

class "0" weight = 1.0

class "1" weight = 1.0

Si especificamos class_weight='balanced', el algoritmo calculará cuántas veces la clase "0" ocurre con más frecuencia que la clase "1". 
Denotaremos este número como N (un número desconocido de veces). Los nuevos pesos de clase se ven así:

class "0" weight = 1.0

class "1" weight = N


La clase rara tendrá un mayor peso.
Los árboles de decisión y los bosques aleatorios también tienen el argumento class_weight.
'''
#Ejercicio

''' 
Aquí hay un código de entrenamiento para un modelo de regresión logística con clases igualmente ponderadas de las lecciones anteriores. 
Equilibra los pesos de clase pasando el valor de argumento adecuado para class_weight. 
Observa cómo cambia el valor F1
'''
try:
    import pandas as pd
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = LogisticRegression(random_state=12345, solver='liblinear',class_weight='balanced')
    model.fit(features_train, target_train)
    predicted_valid = model.predict(features_valid)
    print('F1:', f1_score(target_valid, predicted_valid))
except:print("prueba")
#Exactitud para el árbol de decisión

'''
Primero, debemos verificar si la métrica de exactitud es adecuada para la tarea. Entrenemos al modelo y veamos.
Podemos calcular la exactitud del modelo con la función accuracy_score() . 
Esta toma respuestas y predicciones correctas y devuelve la proporción de respuestas clasificadas correctamente.

Hemos guardado el conjunto de datos con las funciones preparadas en el archivo travel_insurance_us_preprocessed.csv.
'''

#Ejercicio
'''
Entrena el modelo de árbol de decisión calculando el valor de exactitud en el conjunto de validación. 
Para hacerlo, declara la variable predicted_valid, luego guarda el resultado en la variable accuracy_valid.Especifica random_state=12345. Muéstrala en la pantalla.
'''

try:

    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )
    model = DecisionTreeClassifier(random_state=12345)
    model.fit(features_train, target_train)
    predicted_valid = model.predict(features_valid)

    accuracy_valid = accuracy_score(target_valid, predicted_valid)
    print(accuracy_valid)
except: print("prueba")
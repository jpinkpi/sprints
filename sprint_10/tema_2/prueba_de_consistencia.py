#Prueba de consistencia

'''
La proporción de respuestas correctas es del 97%. 

¿Es suficiente? Comparemos con la variable objetivo.

Para evaluar la cordura del modelo, verifiquemos con qué frecuencia la variable objetivo contiene la clase "1" o "0". 
El número de valores únicos se calcula utilizando el método value_counts(), que agrupa exactamente los mismos valores.
'''

#EJERCICIO 1 

'''

1.
Para contar clases en la característica objetivo, utiliza el método value_counts(). Haz las frecuencias relativas (de 0 a 1). La documentación de Pandas te ayudará a hacerlo.
Guarda los valores en la variable class_frequency. Muéstralos en pantalla
Usa el método plot() con el argumento kind='bar'' para trazar un diagrama.

'''

try:
    import pandas as pd

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    class_frequency= data["Claim"].value_counts(normalize=True)
    print(class_frequency)
    class_frequency.plot(kind="bar")
except:print("prueba")

#Ejercicio 2
'''
Analiza las frecuencias de clase de las predicciones del árbol de decisión (la variable predicted_valid).

De manera similar:

Aplica el método value_counts(). Relativiza las frecuencias.
Guarda los valores en la variable class_frequency. Muéstralos en pantalla.
Usa el método plot() con el argumento kind='bar' para trazar un diagrama.
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

    # para hacer que value_counts() funcione,
    # convertimos los resultados a pd.Series
    predicted_valid = pd.Series(model.predict(features_valid))
    class_frequency= predicted_valid.value_counts(normalize=True)
    print(class_frequency)
    class_frequency.plot(kind = "bar")
except:print("prueba")





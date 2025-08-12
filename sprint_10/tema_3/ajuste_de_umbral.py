#Ajuste de umbral
'''
A medida que cambiamos el valor del umbral, también veremos cómo cambian nuestras métricas.
En sklearn, esto se puede investigar utilizando la función predict_proba(), 
que proporciona la probabilidad de que cada observación pertenezca a cada clase posible. 
Esta función toma las características de las observaciones como entrada y devuelve un array de probabilidades para cada clase.


                            probabilities = model.predict_proba(features)

Así se verá el resultado si lo imprimimos: Este modelo genera dos probabilidades para cinco observaciones. La primera columna indica la probabilidad de clase negativa y la segunda indica la probabilidad de clase positiva (las dos probabilidades suman uno).

print(probabilities)
[[0.5795 0.4205]
 [0.6629 0.3371]
 [0.7313 0.2687]
 [0.6728 0.3272]
 [0.5086 0.4914]]


predict_proba() también está disponible en sklearn para árboles de decisión y bosques aleatorios.
'''

#EJERCICIO
'''

Encuentra las probabilidades de clase para la muestra de validación.
Almacena los valores para las probabilidades de clase "1" en la variable probabilities_one_valid. 
Muestra en la pantalla los primeros cinco elementos de la variable (en precódigo).

'''
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = LogisticRegression(random_state=12345, solver='liblinear')
    model.fit(features_train, target_train)
    probabilities_valid = model.predict_proba(features_valid)
    probabilities_one_valid = probabilities_valid[:, 1]
    # < escribe el código aquí  >

    print(probabilities_one_valid[:5])
except:print("prueba")

#Ejercicio 2

'''
.

Pasa por los valores de umbral de 0 a 0.3 en intervalos de 0.02. Encuentra precisión y recall para cada valor del umbral. 
Muestra en pantalla los resultados (en precódigo)

Para crear un bucle con el rango deseado, usamos la función arange() de la librería numpy.
Al igual que la función range(), ésta itera sobre los elementos especificados del rango, 
pero es diferente porque funciona con números fraccionarios además de enteros.

Más adelante aprenderemos más sobre las herramientas de la librería numpy.
'''


try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = LogisticRegression(random_state=12345, solver='liblinear')
    model.fit(features_train, target_train)
    probabilities_valid = model.predict_proba(features_valid)
    probabilities_one_valid = probabilities_valid[:, 1]

    for threshold in np.arange(0, 0.3, 0.02):
        predicted_valid = probabilities_one_valid > threshold
        precision = precision_score(target_valid, predicted_valid)
        recall = recall_score(target_valid, predicted_valid)
        print(
            'Threshold = {:.2f} | Precision = {:.3f}, Recall = {:.3f}'.format(
                threshold, precision, recall
            )
        )
except:print("prueba")    

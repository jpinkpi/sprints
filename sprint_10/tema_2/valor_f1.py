#VALOR F1
'''
Algunas veces, por separado, recall y precisión no son muy informativas. Debes aumentar simultáneamente los valores de ambas métricas... 
¡O podemos recurrir a una nueva métrica que las combine!
Recall y precision evalúan la calidad de las predicciones de la clase positiva desde diferentes ángulos. 
Recall describe qué tan bien comprende un modelo las propiedades de esta clase y es capaz de reconocerla. 
Precisión detecta si un modelo está exagerando el reconocimiento de clase positiva al asignar demasiadas etiquetas positivas.

Ambas métricas son importantes. Las métricas de agregación, una de los cuales es el  valor F1, ayudan a controlarlas simultáneamente. 
Esta es la media armónica de recall y precisión. En F1, 1 significa que la relación de recall a precisión es 1:1.


'''


#EJERCICIO 1
'''
Calcula lo siguiente:

precisión, usando la función precision_score()
recall, usando la función recall_score()
Valor F1, utilizando la fórmula de la lección.
Asigna los valores de las métricas a las variables precision, recall y f1.

Finalmente, muéstralos en pantalla (ya en el precódigo).
'''
try:
    import pandas as pd
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    # < importa las funciones de sklearn.metrics >

    target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
    predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

    precision = precision_score(target, predictions) # <escribe el código aquí  >
    recall = recall_score(target, predictions)# < escribe el código aquí  >
    f1 = 2*precision*recall / (precision + recall)# <escribe el código aquí  >

    print('Recall:', recall)
    print('Precisión:', precision)
    print('Puntuación F1', f1)
except: print("prueba")






#EJERCICIO 2
'''
En el módulo sklearn.metrics, busca la función que calcula el valor F1. Impórtala.

Esta función toma las respuestas y predicciones correctas y devuelve la media armónica de recall y precisión. Muestra los resultados en la pantalla.
'''


try:
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score# < escribe el código aquí >

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = DecisionTreeClassifier(random_state=12345)
    model.fit(features_train, target_train)
    predicted_valid = model.predict(features_valid)
    f1  = f1_score(target_valid, predicted_valid)
    print(f1)
except: print("prueba")
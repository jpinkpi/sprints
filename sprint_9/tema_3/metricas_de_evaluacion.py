#Métricas de evaluación en Scikit-Learn
import sys

'''
Sklearn tiene muchas funciones para calcular métricas, por lo que no tenemos que usar una fórmula para encontrar la exactitud.

Las funciones métricas de la librería sklearn se encuentran en el módulo sklearn.metrics. Para calcular la exactitud utiliza la función accuracy_score().

'''
try:
    from sklearn.metrics import accuracy_score

    #La función toma dos argumentos (las respuestas correctas y las predicciones del modelo) y devuelve el valor de exactitud.
    target = "lalala"
    predictions= "gambling shit"
    accuracy = accuracy_score(target, predictions)
except:print("prueba")

#ejercicio 
'''
¿La puntuación de exactitud difiere entre el conjunto de entrenamiento y el conjunto de prueba? Calcula los valores y muéstralos en la pantalla de la siguiente manera:

Exactitud
Training set: ...
Test set: ...
Guarda las predicciones en las variables train_predictions (para el conjunto de entrenamiento) y test_predictions (para el conjunto de prueba).
'''

try:
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    # < importa la función de cálculo de precisión de la librería  sklearn >

    df = pd.read_csv('/datasets/train_data_us.csv')

    df.loc[df['last_price'] > 113000, 'price_class'] = 1
    df.loc[df['last_price'] <= 113000, 'price_class'] = 0

    features = df.drop(['last_price', 'price_class'], axis=1)
    target = df['price_class']

    model = DecisionTreeClassifier(random_state=12345)

    model.fit(features, target)

    test_df = pd.read_csv('/datasets/test_data_us.csv')

    test_df.loc[test_df['last_price'] > 113000, 'price_class'] = 1
    test_df.loc[test_df['last_price'] <= 113000, 'price_class'] = 0

    test_features = test_df.drop(['last_price', 'price_class'], axis=1)
    test_target = test_df['price_class']

    #< escribe aquí el código para los cálculos del conjunto de entrenamiento >
    train_predictions =  model.predict(features)


    test_predictions  = model.predict(test_features)
    # < escribe aquí el código para los cálculos del conjunto de prueba >

    print('Exactitud')
    print('Training set:', accuracy_score(target, train_predictions)) # < termina el código aquí >)
    print('Test set:', accuracy_score(test_target, test_predictions)) # < termina el código aquí >)

except: print("prueba")
print(sys.version)
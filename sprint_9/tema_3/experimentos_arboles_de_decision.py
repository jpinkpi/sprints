#Experimentos con árboles de decisión
'''
¿Cómo afecta exactamente la profundidad del árbol a la exactitud de la predicción?
Para averiguarlo, vamos a recorrer diferentes niveles de profundidad del árbol en el algoritmo de entrenamiento.
Ahora intentaremos determinar a qué nivel de profundidad del árbol el sobreajuste empieza a afectar a la exactitud de nuestros árboles.
Vamos a aumentar la profundidad máxima del árbol en uno para cada nuevo modelo, comparar los valores de exactitud en los conjuntos de entrenamiento y prueba, y ver dónde empiezan a divergir.

'''

#EJERCICIO 1
'''
Haz que el programa de entrenamiento del árbol de decisión pruebe varias configuraciones del parámetro de profundidad máxima del árbol max_depth. 
El programa tiene que:

Iterar sobre los valores del 1 al 10.
Entrenar modelos en el conjunto de entrenamiento. 
No olvides especificar random_state=54321 al inicializar el constructor del modelo.
Imprimir la puntuación de exactitud de cada modelo en los conjuntos de entrenamiento y de prueba.
'''
try:
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    df = pd.read_csv('/datasets/train_data_us.csv')

    df.loc[df['last_price'] > 113000, 'price_class'] = 1
    df.loc[df['last_price'] <= 113000, 'price_class'] = 0

    features = df.drop(['last_price', 'price_class'], axis=1)
    target = df['price_class']

    test_df = pd.read_csv('/datasets/test_data_us.csv')

    test_df.loc[test_df['last_price'] > 113000, 'price_class'] = 1
    test_df.loc[test_df['last_price'] <= 113000, 'price_class'] = 0

    test_features = test_df.drop(['last_price', 'price_class'], axis=1)
    test_target = test_df['price_class']

    for depth in range(1,11):  # selecciona el rango del hiperparámetro
        # < escribe tu código aquí >
        model= DecisionTreeClassifier(random_state=54321, max_depth=depth)
        model.fit(features, target)
        train_predictions = model.predict(features)
        test_predictions = model.predict(test_features)
        
        print("Exactitud de max_depth igual a", depth)
        print("Conjunto de entrenamiento:", accuracy_score(target,train_predictions)) # calcula la puntuación de accuracy en el conjunto de entrenamiento
        print("Conjunto de prueba:", accuracy_score(test_target, test_predictions))
        
        # calcula la puntuación de accuracy en el conjunto de prueba
        print()


except: print("preuba")

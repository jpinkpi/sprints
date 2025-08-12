#Dataset de Prueba
'''
Ahora es momento poner el modelo a prueba. ¿Cómo comprobamos su conocimiento? Necesitaremos un nuevo conjunto de datos con respuestas conocidas.

Recordemos el código de entrenamiento del modelo:
'''
try:
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    df = pd.read_csv('/datasets/train_data_us.csv')

    df.loc[df['last_price'] > 113000, 'price_class'] = 1
    df.loc[df['last_price'] <= 113000, 'price_class'] = 0

    features = df.drop(['last_price', 'price_class'], axis=1)
    target = df['price_class']

    model = DecisionTreeClassifier(random_state=12345)

    model.fit(features, target)
except:
    print("prueba")

'''
Para probar si nuestro modelo hace predicciones precisas incluso cuando se enfrenta a nuevos datos, vamos a utilizar un nuevo conjunto de datos. 
Ese será el conjunto de datos de prueba.
Nombremos el archivo de datos test_data_us.csv y veamos cómo lo maneja el modelo.
'''

#EJERCICIO 
# Asigna las tres primeras observaciones del conjunto de prueba (/datasets/test_data_us.csv) a la variable test_df. Guarda las funciones utilizadas para la clasificación en la variable test_features.
# Haz una predicción de las respuestas.

# Crea la nueva columna con las respuestas correctas (price_class) y calcula sus valores de la misma manera que lo hiciste para el conjunto de datos principal. 
# Copia esta columna en la variable test_target.
# Imprime las predicciones y las respuestas correctas en la pantalla de la siguiente manera:

# Predicciones: [... ... ...]
# Respuestas correctas: [... ... ...]
# Las matrices de salida tienen que ser numpy.ndarray. 
# El modelo genera este formato de manera predeterminada, así que deja test_predictions como está. Usa el atributo values para tomar los valores de numpy.ndarray desde el objeto serie test_target. 

# Averigüa cuántos errores ha cometido el modelo.

try:
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    # el conjunto de entrenamiento está en el archivo train_data_us.csv 
    df = pd.read_csv('/datasets/train_data_us.csv')

    df.loc[df['last_price'] > 113000, 'price_class'] = 1
    df.loc[df['last_price'] <= 113000, 'price_class'] = 0

    features = df.drop(['last_price', 'price_class'], axis=1)
    target = df['price_class']

    model = DecisionTreeClassifier(random_state=12345)

    model.fit(features, target)

    # < escribe tu codigo aqui >
    test_df = pd.read_csv("/datasets/test_data_us.csv")[:3]

    test_df.loc[test_df['last_price'] > 113000, 'price_class'] = 1
    test_df.loc[test_df['last_price'] <= 113000, 'price_class'] = 0

    test_features = test_df.drop(['last_price', 'price_class'], axis=1)
    test_target = test_df['price_class']
    test_predictions = model.predict(test_features)

    print("Predicciones:", test_predictions)
    print("Respuestas correctas:",test_target.values)
except:
    print("prueba")



#EJERCICIO 2
# Tres ejemplos no son suficientes para saber si el modelo funciona bien o no. 
# Cuenta el número de errores para todo el conjunto de prueba.

# Escribe la función error_count(). El modelo toma las respuestas y predicciones correctas y devuelve el número de discrepancias. 
# Muestra el resultado en pantalla de la siguiente manera (ya en el precódigo):

# Errores: ...

try:
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

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
    test_predictions = model.predict(test_features)

    def error_count(answers, predictions):
        return(answers != predictions).sum()# < función de código aquí  >

    print('Errores:', error_count(test_target, test_predictions))
except:
    print("prueba")




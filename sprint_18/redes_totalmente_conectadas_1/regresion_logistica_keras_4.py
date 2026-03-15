#Regresion Logistica en Keras 
'''
Ya escribimos una regresión lineal en Keras, es hora de escribir una regresión logística.

Para obtener una regresión logística, solo necesitamos cambiar el código de regresión lineal en dos lugares:

1.Aplica la función de activación a la capa totalmente conectada:
keras.layers.Dense(units=1, input_dim=features_train.shape[1], 
                                     activation='sigmoid')

2.Cambia la función de pérdida del ECM a binary_crossentropy:
model.compile(loss='binary_crossentropy', optimizer='sgd')
'''

#Ejercicios

#1
'''
1.

Entrena la regresión logística utilizando los datos cargados en el precódigo. 
Establece el número de épocas en cinco.

Para imprimir el progreso del entrenamiento, especifica verbose=2 para fit().
'''
try:
    import pandas as pd
    from tensorflow import keras

    df = pd.read_csv('/datasets/train_data_n.csv')
    df['target'] = (df['target'] > df['target'].median()).astype(int)
    features_train = df.drop('target', axis=1)
    target_train = df['target']

    df_val = pd.read_csv('/datasets/test_data_n.csv')
    df_val['target'] = (df_val['target'] > df['target'].median()).astype(int)
    features_valid = df_val.drop('target', axis=1)
    target_valid = df_val['target']
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=1, input_dim=features_train .shape[1],activation='sigmoid'))
    model.compile(loss= 'binary_crossentropy', optimizer='sgd')
    model.fit(features_train, target_train, validation_data=(features_valid, target_valid), verbose=2, epochs=5)
    # < escribe tu código aquí >
except:print("Prueba")

#Ejercicio 2

'''
2.

Encuentra la exactitud del modelo utilizando el conjunto de validación. 
Calcula las predicciones con la función predict(), al igual que en sklearn. 
Sigmoide devolverá números de 0 a 1. Convierte los números resultantes 
en clases en comparación con 0.5.

Imprime el valor de exactitud (en el precódigo). 
Especifica verbose=0 para deshacerte de la salida del progreso de entrenamiento.
'''

try:
    import pandas as pd
    from tensorflow import keras
    from sklearn.metrics import accuracy_score


    df = pd.read_csv('/datasets/train_data_n.csv')
    df['target'] = (df['target'] > df['target'].median()).astype(int)
    features_train = df.drop('target', axis=1)
    target_train = df['target']

    df_val = pd.read_csv('/datasets/test_data_n.csv')
    df_val['target'] = (df_val['target'] > df['target'].median()).astype(int)
    features_valid = df_val.drop('target', axis=1)
    target_valid = df_val['target']

    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units=1, input_dim=features_train.shape[1], activation='sigmoid'
        )
    )
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.fit(
        features_train,
        target_train,
        epochs=5,
        verbose=0,
        validation_data=(features_valid, target_valid),
    )

    predictions = model.predict(features_valid) > 0.5
    # < escribe tu código aquí  >
    print('Exactitud:',accuracy_score(target_valid, predictions)) # < escribe tu código aquí  >)
except:print("Prueba")

#Ejercicio 3
'''
3.

Por lo general, entrenar una red neuronal lleva mucho tiempo. 
Modifiquemos el modelo para que podamos rastrear su calidad en cada época: 
agrega el parámetro metrics=['acc'] (exactitud) al método compile().

Entrena el modelo, usando diez épocas para mejorar la exactitud.
'''
try:
    import pandas as pd
    from tensorflow import keras
    from sklearn.metrics import accuracy_score


    df = pd.read_csv('/datasets/train_data_n.csv')
    df['target'] = (df['target'] > df['target'].median()).astype(int)
    features_train = df.drop('target', axis=1)
    target_train = df['target']

    df_val = pd.read_csv('/datasets/test_data_n.csv')
    df_val['target'] = (df_val['target'] > df['target'].median()).astype(int)
    features_valid = df_val.drop('target', axis=1)
    target_valid = df_val['target']

    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units=1, input_dim=features_train.shape[1], activation='sigmoid'
        )
    )
    model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['acc'])
    model.fit(
        features_train,
        target_train,
        epochs=10,
        verbose=2,
        validation_data=(features_valid, target_valid))

    predictions = model.predict(features_valid) > 0.5
    # < escribe tu código aquí  >
    print('Exactitud:',accuracy_score(target_valid, predictions)) # < escribe tu código aquí  >)
except:print("Prueba")
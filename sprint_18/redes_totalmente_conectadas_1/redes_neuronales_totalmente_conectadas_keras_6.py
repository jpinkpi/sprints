#Redes Neuronales Totalmente Conectadas en Keras
'''
En Keras, las capas totalmente conectadas se pueden crear llamando a Dense(). 
¿Qué debemos hacer para construir una red multicapa totalmente conectada?

¡Agregar una capa totalmente conectada varias veces! 
Más capas representan un modelo más complejo.

'''
#EJERCICIO 
'''
Agrega otra capa a la red neuronal. 
Deja que la primera capa oculta tenga diez neuronas (units) con activación sigmoide. 
La segunda capa de salida tendrá una neurona con un sigmoide y 
la consideraremos en la probabilidad de la clase "1".

Usa diez épocas para entrenar la red neuronal. Imprime el progreso del entrenamiento.
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

    model = keras.models.Sequential() # < escribe tu código aquí  >
    model.add(keras.layers.Dense(units=10, activation="sigmoid", input_dim=features_train.shape[1]))# < escribe tu código aquí  >
    model.add(keras.layers.Dense(1,activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])

    model.fit(features_train, target_train, epochs=10, verbose=2,
            validation_data=(features_valid, target_valid))
except:print("Prueba")

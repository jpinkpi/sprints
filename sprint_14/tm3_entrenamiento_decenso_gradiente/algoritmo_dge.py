#Algoritmo DGE 
'''
Escribe un algoritmo DGE para un modelo de regresión lineal.
En esta lección aprenderemos cómo pasar hiperparámetros a un modelo. Necesitamos declarar la clase del modelo y crear el método "inicializador de clase" (__init__):
'''

class SGDLinearRegression:
    def __init__(self):
        ...

#Agrega un hiperparámetro step_size al inicializador de clase:

class SGDLinearRegression:
    def __init__(self, step_size):
        ...
#Ahora podemos pasar el tamaño del paso al modelo al crear una clase:

# puedes elegir el tamaño del paso de forma arbitraria
model = SGDLinearRegression(0.01)
#Guarda el tamaño del paso como un atributo:

class SGDLinearRegression:
    def __init__(self, step_size):
        self.step_size = step_size


#ejercicios 

'''
1.

Inicia el desarrollo del algoritmo DGE con un código ficticio. 
La descarga de datos y la ejecución de los algoritmos ya está en precódigo. 
Termina el código de clase del modelo:

Agrega los hiperparámetros epochs y batch_size.

    Agrégalos al inicializador de clase en ese orden y guárdalos en los atributos self.epochs y self.batch_size.
    En fit(), establece los pesos iniciales (w) en cero.
    En predict(), escribe la fórmula para el cálculo de predicción.
    Hemos elegido la métrica R2. Muestra sus valores en pantalla (en precódigo).
'''

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


try: 
    data_train = pd.read_csv('/datasets/train_data_n.csv')
    features_train = data_train.drop(['target'], axis=1)
    target_train = data_train['target']

    data_test = pd.read_csv('/datasets/test_data_n.csv')
    features_test = data_test.drop(['target'], axis=1)
    target_test = data_test['target']


    class SGDLinearRegression:
        def __init__(self, step_size, epochs, batch_size):
            # < escribe tu código aquí >)):
            self.step_size = step_size
            self.epochs = epochs
            self.batch_size = batch_size
            # < escribe tu código aquí >)

        def fit(self, train_features, train_target):
            X = np.concatenate(
                (np.ones((train_features.shape[0], 1)), train_features), axis=1
            )
            y = train_target
            w = np.zeros(X.shape[1]) # < escribe tu código aquí >)

        # escribirás el algoritmo DGE aquí en los próximos ejercicios

            self.w = w[1:]
            self.w0 = w[0]

        def predict(self, test_features):
            return test_features.dot(self.w) + self.w0

    # ya se han superado los parámetros de entrenamiento adecuados
    model = SGDLinearRegression(0.01, 1, 200)
    model.fit(features_train, target_train)
    pred_train = model.predict(features_train)
    pred_test = model.predict(features_test)
    print(r2_score(target_train, pred_train).round(5))
    print(r2_score(target_test, pred_test).round(5))

except: print("Prueba, no tenemos bases de datos")

'''
2.

Agrega bucles por épocas y lotes, 
sin tener en cuenta por ahora el paso de gradiente negativo.

Necesitas:

Encontrar el número de lotes;
Encontrar el comienzo del lote i: el índice del primer elemento;
Encontrar el final del lote i: el comienzo del siguiente lote.
Muestra en la pantalla los valores R2 (en precódigo).
'''

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

try:
    data_train = pd.read_csv('/datasets/train_data_n.csv')
    features_train = data_train.drop(['target'], axis=1)
    target_train = data_train['target']

    data_test = pd.read_csv('/datasets/test_data_n.csv')
    features_test = data_test.drop(['target'], axis=1)
    target_test = data_test['target']


    class SGDLinearRegression:
        def __init__(self, step_size, epochs, batch_size):
            self.step_size = step_size
            self.epochs = epochs
            self.batch_size = batch_size

        def fit(self, train_features, train_target):
            X = np.concatenate(
                (np.ones((train_features.shape[0], 1)), train_features), axis=1
            )
            y = train_target
            w = np.zeros(X.shape[1])
            
            for _ in range(self.epochs):
                batches_count =  int(np.ceil(X.shape[0] / self.batch_size)) # < write code here >
                for i in range(batches_count):
                    begin = i * self.batch_size # < write code here >
                    end = min(begin + self.batch_size, X.shape[0])# < write code here >
                    X_batch = X[begin:end, :]
                    y_batch = y[begin:end]
                    
                    # escribirás el paso de gradiente negativo aquí en el próximo ejercicio

            self.w = w[1:]
            self.w0 = w[0]

        def predict(self, test_features):
            return test_features.dot(self.w) + self.w0
        
    model = SGDLinearRegression(0.01, 10, 100)
    model.fit(features_train, target_train)
    pred_train = model.predict(features_train)
    pred_test = model.predict(features_test)
    print(r2_score(target_train, pred_train).round(5))
    print(r2_score(target_test, pred_test).round(5))
except: print("Prueba, no tenemos bases de datos")

'''
3.

Termina de escribir el algoritmo DGE:

Encuentra el gradiente para el lote;
Crea un paso a lo largo del gradiente negativo para los pesos.
Muestra en la pantalla el resultado del R2 (en precódigo).

'''
try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score


    data_train = pd.read_csv('/datasets/train_data_n.csv')
    features_train = data_train.drop(['target'], axis=1)
    target_train = data_train['target']

    data_test = pd.read_csv('/datasets/test_data_n.csv')
    features_test = data_test.drop(['target'], axis=1)
    target_test = data_test['target']


    class SGDLinearRegression:
        def __init__(self, step_size, epochs, batch_size):
            self.step_size = step_size
            self.epochs = epochs
            self.batch_size = batch_size
        
        def fit(self, train_features, train_target):
            X = np.concatenate(
                (np.ones((train_features.shape[0], 1)), train_features), axis=1
            )
            y = train_target
            w = np.zeros(X.shape[1])
            
            for _ in range(self.epochs):
                batches_count = X.shape[0] // self.batch_size
                for i in range(batches_count):
                    begin = i * self.batch_size
                    end = (i + 1) * self.batch_size
                    X_batch = X[begin:end, :]
                    y_batch = y[begin:end]
                    
                    gradient = 2 * X_batch.T.dot(X_batch.dot(w)- y_batch) / X_batch.shape[0] # < escribe tu código aquí >
                    
                    w -= self.step_size * gradient # < escribe tu código aquí >

            self.w = w[1:]
            self.w0 = w[0]

        def predict(self, test_features):
            return test_features.dot(self.w) + self.w0
        
    model = SGDLinearRegression(0.01, 10, 100)
    model.fit(features_train, target_train)
    pred_train = model.predict(features_train)
    pred_test = model.predict(features_test)
    print(r2_score(target_train, pred_train).round(5))
    print(r2_score(target_test, pred_test).round(5))
except: print("Prueba, no tenemos bases de datos")
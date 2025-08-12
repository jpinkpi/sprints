#Entranemiento de una Regresion Lineal

'''
Volvamos a nuestra función objetivo.
Hasta ahora hemos expresado la función de pérdida de ECM así:

w = argmin₍w₎ (ECM(Xw, y))

También podemos expresar esta fórmula en forma matricial:

w = (XᵀX)⁻¹Xᵀy

Donde:

-w representa el vector de pesos de regresión;
-X representa la matriz de observaciones con las características;

-y representa el vector de columna de observaciones con el objetivo.

¿Cómo podemos entender esta fórmula en pasos?
La matriz de características transpuesta se multiplica por sí misma.
Se calcula la matriz inversa a ese resultado.
La matriz inversa se multiplica por la matriz de características transpuesta.
El resultado se multiplica por el vector de los valores objetivo.
'''

#Ejercicio 1 
'''
.

Escribe código para entrenar y probar el modelo (en el conjunto de entrenamiento) y, 
en lugar del modelo, crea un código dummy, un código simple que no signifique nada. 
Servirá como marcador de posición y te ayudará a asegurarte de que tu código funcione 
antes de comenzar a escribir el modelo.

Para crear un código dummy:

crea la clase LinearRegression;
escribe el método fit() que:

acepte las características y la variable objetivo como entrada;
cree los atributos w y w0 y los establezca en None.
escribe un método predict() que tome características y devuelva una predicción de 0 
para todas las observaciones.
Crea un modelo, guárdalo en la variable model y entrénalo. 
Encuentra sus predicciones en el conjunto de entrenamiento y guárdalas 
en la variable predictions. Muestra en la pantalla el valor de la métrica R2. 
Este se calcula usando el ECM. Al comparar su valor con cero, 
averigua si el modelo pasa la prueba de cordura o no.
'''
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score# < escribe tu código aquí >

columns = ['bedrooms', 'total area', 'kitchen', 'living area', 'floor', 'total floors', 'price']

data = pd.DataFrame([
    [1, 38.5, 6.9, 18.9, 3, 5, 84000],
    [1, 38.0, 8.5, 19.2, 9, 17, 70000],
    [1, 34.7, 10.3, 19.8, 1, 9, 102000],
    [1, 45.9, 11.1, 17.5, 11, 23, 126000],
    [1, 42.4, 10.0, 19.9, 6, 14, 118000],
    [1, 46.0, 10.2, 20.5, 3, 12, 162000],
    [2, 77.7, 13.2, 39.3, 3, 17, 148000],
    [2, 69.8, 11.1, 31.4, 12, 23, 144000],
    [2, 78.2, 19.4, 33.2, 4, 9, 136000],
    [2, 55.5, 7.8, 29.6, 1, 25, 186000],
    [2, 74.3, 16.0, 34.2, 14, 17, 212000],
    [2, 78.3, 12.3, 42.6, 23, 23, 170000],
    [2, 74.0, 18.1, 49.0, 8, 9, 120000],
    [2, 91.4, 20.1, 60.4, 2, 10, 144000],
    [3, 85.0, 17.8, 56.1, 14, 14, 250000],
    [3, 79.8, 9.8, 44.8, 9, 10, 264000],
    [3, 72.0, 10.2, 37.3, 7, 9, 302000],
    [3, 95.3, 11.0, 51.5, 15, 23, 196000],
    [3, 69.3, 8.5, 39.3, 4, 9, 228000],
    [3, 89.8, 11.2, 58.2, 24, 25, 326000],
], columns=columns)

features = data.drop('price', axis=1)
target = data['price']

class LinearRegression:
    def fit(self,train_features, train_target):
        self.w = None#
        self.w0 = None
    def predict(self,test_features):
        return np.zeros(test_features.shape[0])
        

model = LinearRegression() # < escribe tu código aquí >
# < escribe tu código aquí >
predictions = model.predict(features)
print(r2_score(target, predictions))


#EJERCICIO 2
'''
Escribe la función predict() para calcular las predicciones de regresión lineal.

Actualiza el código ficticio para los parámetros w y w0 en la función fit().  
Rellena el vector w con ceros usando una longitud igual al número de características (train_features.shape[1]) 
y asigna el valor medio de la variable objetivo a w0.
'''

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

columns = ['bedrooms', 'total area', 'kitchen', 'living area', 'floor', 'total floors', 'price']

data = pd.DataFrame([
    [1, 38.5, 6.9, 18.9, 3, 5, 84000],
    [1, 38.0, 8.5, 19.2, 9, 17, 70000],
    [1, 34.7, 10.3, 19.8, 1, 9, 102000],
    [1, 45.9, 11.1, 17.5, 11, 23, 126000],
    [1, 42.4, 10.0, 19.9, 6, 14, 118000],
    [1, 46.0, 10.2, 20.5, 3, 12, 162000],
    [2, 77.7, 13.2, 39.3, 3, 17, 148000],
    [2, 69.8, 11.1, 31.4, 12, 23, 144000],
    [2, 78.2, 19.4, 33.2, 4, 9, 136000],
    [2, 55.5, 7.8, 29.6, 1, 25, 186000],
    [2, 74.3, 16.0, 34.2, 14, 17, 212000],
    [2, 78.3, 12.3, 42.6, 23, 23, 170000],
    [2, 74.0, 18.1, 49.0, 8, 9, 120000],
    [2, 91.4, 20.1, 60.4, 2, 10, 144000],
    [3, 85.0, 17.8, 56.1, 14, 14, 250000],
    [3, 79.8, 9.8, 44.8, 9, 10, 264000],
    [3, 72.0, 10.2, 37.3, 7, 9, 302000],
    [3, 95.3, 11.0, 51.5, 15, 23, 196000],
    [3, 69.3, 8.5, 39.3, 4, 9, 228000],
    [3, 89.8, 11.2, 58.2, 24, 25, 326000],
], columns=columns)

features = data.drop('price', axis=1)
target = data['price']

class LinearRegression:
    def fit(self, train_features, train_target):
        self.w = np.zeros(train_features.shape[1])
        self.w0 = train_target.mean()

    def predict(self, test_features):
        return test_features.dot(self.w) + self.w0
    
model = LinearRegression()
model.fit(features, target)
predictions = model.predict(features)
print(r2_score(target, predictions))


#EJERCICIO 3
'''
Ahora hemos agregado al precódigo la versión abreviada de la fórmula de regresión lineal 
de la que hablamos antes: una columna de unos al comienzo del conjunto de entrenamiento.

Termina el código para calcular w usando la fórmula de minimización de ECM.  
Luego regresa a la versión original de los parámetros w y w0 (en precódigo).
'''
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

columns = ['bedrooms', 'total area', 'kitchen', 'living area', 'floor', 'total floors', 'price']

data = pd.DataFrame([
    [1, 38.5, 6.9, 18.9, 3, 5, 84000],
    [1, 38.0, 8.5, 19.2, 9, 17, 70000],
    [1, 34.7, 10.3, 19.8, 1, 9, 102000],
    [1, 45.9, 11.1, 17.5, 11, 23, 126000],
    [1, 42.4, 10.0, 19.9, 6, 14, 118000],
    [1, 46.0, 10.2, 20.5, 3, 12, 162000],
    [2, 77.7, 13.2, 39.3, 3, 17, 148000],
    [2, 69.8, 11.1, 31.4, 12, 23, 144000],
    [2, 78.2, 19.4, 33.2, 4, 9, 136000],
    [2, 55.5, 7.8, 29.6, 1, 25, 186000],
    [2, 74.3, 16.0, 34.2, 14, 17, 212000],
    [2, 78.3, 12.3, 42.6, 23, 23, 170000],
    [2, 74.0, 18.1, 49.0, 8, 9, 120000],
    [2, 91.4, 20.1, 60.4, 2, 10, 144000],
    [3, 85.0, 17.8, 56.1, 14, 14, 250000],
    [3, 79.8, 9.8, 44.8, 9, 10, 264000],
    [3, 72.0, 10.2, 37.3, 7, 9, 302000],
    [3, 95.3, 11.0, 51.5, 15, 23, 196000],
    [3, 69.3, 8.5, 39.3, 4, 9, 228000],
    [3, 89.8, 11.2, 58.2, 24, 25, 326000],
], columns=columns)

features = data.drop('price', axis=1)
target = data['price']

class LinearRegression:
    def fit(self, train_features, train_target):
        X = np.concatenate((np.ones((train_features.shape[0], 1)), train_features), axis=1)
        y = train_target
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)# < escribe el código aquí >
        self.w = w[1:]
        self.w0 = w[0]

    def predict(self, test_features):
        return test_features.dot(self.w) + self.w0
    
model = LinearRegression()
model.fit(features, target)
predictions = model.predict(features)
print(r2_score(target, predictions))
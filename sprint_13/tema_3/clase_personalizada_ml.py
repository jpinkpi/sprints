#Creación de una clase personalizada de ML

'''
Ahora pondremos en práctica lo que hemos aprendido hasta ahora a través de la creación de una clase de machine learning personalizada.
Aunque las librerías del tipo sklearn disponen de una serie de algoritmos de machine learning muy útiles, 
de vez en cuando puedes encontrarte con la necesidad de escribir y personalizar un algoritmo por tu propia cuenta. 
Es posible que te sorprenda, pero gracias a esos temas fundamentales de álgebra lineal, que ya hemos estado trabajando, podemos hacer precisamente eso.

Cuando importamos un algoritmo de machine learning, por ejemplo, DecisionTreeClassifier() o RandomForestRegressor(), 
en realidad importamos un tipo de datos que en Python se llama clase. Una clase funciona como una plantilla de código. 
Cuando creamos un objeto mediante una clase, el objeto adopta los métodos y atributos de esa clase. 
No es la primera vez que usamos métodos y atributos, así que deberías recordar que son como las funciones y las variables, 
salvo que se adjuntan a un objeto por medio del caracter .
'''
train_features = "hola"
train_labels = "adios"

try:
    from sklearn.ensemble import RandomForestRegressor

    # aquí estamos creando un objeto usando la clase RandomForestRegressor
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # aquí estamos llamando al método .fit() de nuestra clase
    rf.fit(train_features, train_labels)
except: print("Prueba")

'''
Ahora vamos a crear una clase personalizada, ConstantRegression(), para elaborar un modelo de regresión 
que prediga las respuestas basándose en los valores medios de la variable objetivo del dataset de entrenamiento.

Para crear una nueva clase, especifica la palabra clave class seguida del nombre de la clase. 
Ten en cuenta la convención de nomenclatura para las clases: LetrasMayúsculasSinEspacios:
'''

#class ConstantRegression():
    # contenido de la clase con una indentación de cuatro espacios
    # ...


'''
Para entrenar el modelo, vamos a utilizar el método fit().  Cuando escribimos métodos de clase, el primer parámetro es siempre **self**. self es una variable que se refiere al objeto que llamará al método. 
Si nos olvidamos de self, nuestro código no funcionará. Los otros dos parámetros que necesitamos son train_features y train_target, igual que en sklearn.
'''

#class ConstantRegression():
   # def fit(self, train_features, train_target):
       # contenido de la función con una indentación de 4+4 

'''
Al realizar el entrenamiento, debemos guardar el valor medio del objetivo.
Para crear la nueva mean de atributos, añade self. al principio del nombre de la variable. 
Al igual que con nuestro método, esto indica que la variable forma parte del objeto de la clase:
'''

class ConstantRegression():
    def fit(self, train_features, train_target):
        self.mean = train_target.mean()

'Por último, vamos a crear el método predict() para predecir la respuesta, que es la media guardada:'
class ConstantRegression():
    def fit(self, train_features, train_target):
        self.mean = train_target.mean()

    def predict(self, new_features):
        answer = pd.Series(self.mean, index=new_features.index)
        return answer
    

'Vamos a ejecutar el modelo con nuestros datos inmobiliarios, donde el objetivo es el precio:'
import pandas as pd
from scipy.spatial import distance

columns = [
    'bedrooms',
    'total area',
    'kitchen',
    'living area',
    'floor',
    'total floors',
    'price',
]

df_train = pd.DataFrame(
    [
        [1, 38.5, 6.9, 18.9, 3, 5, 4200000],
        [1, 38.0, 8.5, 19.2, 9, 17, 3500000],
        [1, 34.7, 10.3, 19.8, 1, 9, 5100000],
        [1, 45.9, 11.1, 17.5, 11, 23, 6300000],
        [1, 42.4, 10.0, 19.9, 6, 14, 5900000],
        [1, 46.0, 10.2, 20.5, 3, 12, 8100000],
        [2, 77.7, 13.2, 39.3, 3, 17, 7400000],
        [2, 69.8, 11.1, 31.4, 12, 23, 7200000],
        [2, 78.2, 19.4, 33.2, 4, 9, 6800000],
        [2, 55.5, 7.8, 29.6, 1, 25, 9300000],
        [2, 74.3, 16.0, 34.2, 14, 17, 10600000],
        [2, 78.3, 12.3, 42.6, 23, 23, 8500000],
        [2, 74.0, 18.1, 49.0, 8, 9, 6000000],
        [2, 91.4, 20.1, 60.4, 2, 10, 7200000],
        [3, 85.0, 17.8, 56.1, 14, 14, 12500000],
        [3, 79.8, 9.8, 44.8, 9, 10, 13200000],
        [3, 72.0, 10.2, 37.3, 7, 9, 15100000],
        [3, 95.3, 11.0, 51.5, 15, 23, 9800000],
        [3, 69.3, 8.5, 39.3, 4, 9, 11400000],
        [3, 89.8, 11.2, 58.2, 24, 25, 16300000],
    ],
    columns=columns,
)

train_features = df_train.drop('price', axis=1)
train_target = df_train['price']

df_test = pd.DataFrame(
    [
        [1, 36.5, 5.9, 17.9, 2, 7, 3900000],
        [2, 71.7, 12.2, 34.3, 5, 21, 7100000],
        [3, 88.0, 18.1, 58.2, 17, 17, 12100000],
    ],
    columns=columns,
)

test_features = df_test.drop('price', axis=1)


class ConstantRegression:
    def fit(self, train_features, train_target):
        self.mean = train_target.mean()

    def predict(self, new_features):
        answer = pd.Series(self.mean, index=new_features.index)
        return answer


model = ConstantRegression()
model.fit(train_features, train_target)
test_predictions = model.predict(test_features)
print(test_predictions)

#Ejercicio 1
'''
Crea la clase NearestNeighborClassifier() para el modelo de clasificación.  
En este ejercicio nos ocuparemos únicamente del entrenamiento. Las predicciones las dejaremos para el siguiente.

Añade el método fit() a la clase. En el caso del algoritmo de vecinos más cercanos, fit() guardará todo el conjunto de entrenamiento.

Guarda:

-las características del conjunto de entrenamiento en self.train_features
-el objetivo en self.train_target


Los atributos pueden tener los mismos nombres que los parámetros del modelo.

Entrena el modelo y muestra sus atributos (en precódigo).
'''

import numpy as np
import pandas as pd
from scipy.spatial import distance

columns = [
    "bedrooms",
    "total area",
    "kitchen",
    "living area",
    "floor",
    "total floors",
]

df_train = pd.DataFrame(
    [
        [1.0, 38.5, 6.9, 18.9, 3.0, 5.0],
        [1.0, 38.0, 8.5, 19.2, 9.0, 17.0],
        [1.0, 34.7, 10.3, 19.8, 1.0, 9.0],
        [1.0, 45.9, 11.1, 17.5, 11.0, 23.0],
        [1.0, 42.4, 10.0, 19.9, 6.0, 14.0],
        [1.0, 46.0, 10.2, 20.5, 3.0, 12.0],
        [2.0, 77.7, 13.2, 39.3, 3.0, 17.0],
        [2.0, 69.8, 11.1, 31.4, 12.0, 23.0],
        [2.0, 78.2, 19.4, 33.2, 4.0, 9.0],
        [2.0, 55.5, 7.8, 29.6, 1.0, 25.0],
        [2.0, 74.3, 16.0, 34.2, 14.0, 17.0],
        [2.0, 78.3, 12.3, 42.6, 23.0, 23.0],
        [2.0, 74.0, 18.1, 49.0, 8.0, 9.0],
        [2.0, 91.4, 20.1, 60.4, 2.0, 10.0],
        [3.0, 85.0, 17.8, 56.1, 14.0, 14.0],
        [3.0, 79.8, 9.8, 44.8, 9.0, 10.0],
        [3.0, 72.0, 10.2, 37.3, 7.0, 9.0],
        [3.0, 95.3, 11.0, 51.5, 15.0, 23.0],
        [3.0, 69.3, 8.5, 39.3, 4.0, 9.0],
        [3.0, 89.8, 11.2, 58.2, 24.0, 25.0],
    ],
    columns=columns,
)


train_features = df_train.drop("bedrooms", axis=1)
train_target = df_train["bedrooms"]

df_test = pd.DataFrame(
    [
        [1, 36.5, 5.9, 17.9, 2, 7],
        [2, 71.7, 12.2, 34.3, 5, 21],
        [3, 88.0, 18.1, 58.2, 17, 17],
    ],
    columns=columns,
)

test_features = df_test.drop("bedrooms", axis=1)

class NearestNeighborClassifier:
    def fit(self, train_features, train_target):
        self.train_features = train_features
        self.train_target = train_target# < escribe tu código aquí >

model = NearestNeighborClassifier()
model.fit(train_features, train_target)
print(model.train_features.head())
print(model.train_target.head())

#EJERCICIO 2

'''
Añade predict() a la clase NearestNeighborClassifier(). Utiliza la función nearest_neighbor_predict() que ya hemos creado como plantilla.

Obtén predicciones sobre el número de habitaciones. Observa que las respuestas reales se encuentran en la primera columna de df_test del precódigo. Muestra los resultados.
'''

import numpy as np
import pandas as pd
from scipy.spatial import distance

columns = [
    'bedrooms',
    'total area',
    'kitchen',
    'living area',
    'floor',
    'total floors',
]

df_train = pd.DataFrame(
    [
        [1.0, 38.5, 6.9, 18.9, 3.0, 5.0],
        [1.0, 38.0, 8.5, 19.2, 9.0, 17.0],
        [1.0, 34.7, 10.3, 19.8, 1.0, 9.0],
        [1.0, 45.9, 11.1, 17.5, 11.0, 23.0],
        [1.0, 42.4, 10.0, 19.9, 6.0, 14.0],
        [1.0, 46.0, 10.2, 20.5, 3.0, 12.0],
        [2.0, 77.7, 13.2, 39.3, 3.0, 17.0],
        [2.0, 69.8, 11.1, 31.4, 12.0, 23.0],
        [2.0, 78.2, 19.4, 33.2, 4.0, 9.0],
        [2.0, 55.5, 7.8, 29.6, 1.0, 25.0],
        [2.0, 74.3, 16.0, 34.2, 14.0, 17.0],
        [2.0, 78.3, 12.3, 42.6, 23.0, 23.0],
        [2.0, 74.0, 18.1, 49.0, 8.0, 9.0],
        [2.0, 91.4, 20.1, 60.4, 2.0, 10.0],
        [3.0, 85.0, 17.8, 56.1, 14.0, 14.0],
        [3.0, 79.8, 9.8, 44.8, 9.0, 10.0],
        [3.0, 72.0, 10.2, 37.3, 7.0, 9.0],
        [3.0, 95.3, 11.0, 51.5, 15.0, 23.0],
        [3.0, 69.3, 8.5, 39.3, 4.0, 9.0],
        [3.0, 89.8, 11.2, 58.2, 24.0, 25.0],
    ],
    columns=columns,
)


train_features = df_train.drop('bedrooms', axis=1)
train_target = df_train['bedrooms']

df_test = pd.DataFrame(
    [
        [1, 36.5, 5.9, 17.9, 2, 7],
        [2, 71.7, 12.2, 34.3, 5, 21],
        [3, 88.0, 18.1, 58.2, 17, 17],
    ],
    columns=columns,
)

test_features = df_test.drop('bedrooms', axis=1)


def nearest_neighbor_predict(train_features, train_target, new_features):
    distances = []
    for i in range(train_features.shape[0]):
        vector = train_features.loc[i].values
        distances.append(distance.euclidean(new_features, vector))
    best_index = np.array(distances).argmin()
    return train_target.loc[best_index]


class NearestNeighborClassifier():
    def fit(self, train_features, train_target):
        self.train_features = train_features
        self.train_target = train_target

    def predict(self, new_features):
        values = []
        for i in range(new_features.shape[0]):
            test_vector = new_features.loc[i]
            values.append(nearest_neighbor_predict(self.train_features,self.train_target, test_vector))
        return pd.Series(values)
           # <escribe el código aquí>

model = NearestNeighborClassifier()
model.fit(train_features, train_target)
new_predictions = model.predict(test_features)
print(new_predictions)
#Regresión Logística
'''
El modelo LogisticRegression se puede encontrar en el módulo linear_model de la librería scikit-learn. Impórtalo:

from sklearn.linear_model import LogisticRegression
Almacena el modelo en una variable y especifica los hiperparámetros. Para obtener uniformidad en los resultados, establece random_state en 54321.

También necesitamos especificar un solver, una versión del algoritmo que determine qué tan ajustada está exactamente la curva. 
Por lo general, producen resultados similares. Usaremos el solver 'liblinear' porque es el más general. Funciona bien para conjuntos de datos pequeños con muchas características.
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
model = LogisticRegression(random_state=54321, solver='liblinear')

'''Entrena el modelo llamando al método **fit().

model.fit(features, target)
Llama al método score() para mostrar la exactitud del modelo:

model.score(features, target)
'''

#EJERCICIO 1

'''
Entrena un modelo de regresión logística en el conjunto de entrenamiento, luego calcula el valor de accuracy tanto en el conjunto de entrenamiento como en el conjunto de validación.
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\josep\Downloads\moved_train_data_us.csv")

df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0

df_train, df_valid = train_test_split(df, test_size=.25, random_state=54321) # segmenta el 25% de los datos para hacer el conjunto de validación

features_train = df_train.drop(['last_price', 'price_class'], axis=1)
target_train = df_train['price_class']
features_valid = df_valid.drop(['last_price', 'price_class'], axis=1)
target_valid = df_valid['price_class']

model =  LogisticRegression(random_state=54321, solver="liblinear")# inicializa el constructor de regresión logística con los parámetros random_state=54321 y solver='liblinear'
model.fit(features_train, target_train) # entrena el modelo en el conjunto de entrenamiento
score_train = model.score(features_train, target_train) # calcula la puntuación de accuracy en el conjunto de entrenamiento
score_valid = model.score(features_valid, target_valid) # calcula la puntuación de accuracy en el conjunto de validación

print("Accuracy del modelo de regresión logística en el conjunto de entrenamiento:", score_train)
print("Accuracy del modelo de regresión logística en el conjunto de validación:", score_valid)
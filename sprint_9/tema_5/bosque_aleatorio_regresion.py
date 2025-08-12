#Bosque aleatorio de regresión
'''
Si lo piensas, bosque aleatorio es un nombre lindo... En fin, sigamos con el entrenamiento de un modelo de bosque aleatorio para regresión.
El bosque aleatorio de regresión no difiere mucho del de clasificación.
Lo que hace es entrenar a un grupo de árboles independientes y luego promedia sus respuestas para tomar una decisión.
'''

#EJERCICIO
'''
Selecciona 25% de los datos para la muestra de validación (prueba), lo demás será para el entrenamiento.
Extrae características y objetivos para el entrenamiento y la validación. Para conseguir objetivos, selecciona la columna 'last_price' y divídela entre 100000. La característica son todas las columnas excepto por la columna 'last_price'.
Entrena modelos de bosque aleatorio para el problema de regresión:
con el número de árboles: de 10 a 50, en intervalos de 10,
con una profundidad máxima de 1 a 10.
Para cada modelo, calcula la RECM en el conjunto de validación y guárdalo en la variable error.


Para calcular la métrica RECM, toma el valor de la raíz cuadrada del ECM:

mean_squared_error(target_valid, predictions_valid)**0.5
    
El código se puede ejecutar durante cerca de un minuto. Esto es normal porque estás entrenando 50 modelos.
'''

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"C:\Users\josep\Downloads\moved_train_data_us.csv")

df_train, df_valid = train_test_split(df, random_state=54321, test_size = .25)# haz la división de datos para el entrenamiento y la validación

features_train = df_train.drop(["last_price"], axis=1)# extrae las características de entrenamiento
target_train = df_train["last_price"]/100000# extrae los objetivos de entrenamiento
features_valid = df_valid.drop(["last_price"], axis=1)# extrae las características para la validación
target_valid = df_valid["last_price"]/100000 # extrae los objetivos de validación

best_error = 10000 # configura el inicio de RECM
best_est = 0
best_depth = 0
for est in range(10, 51, 10):
    for depth in range (1, 11):
        model = RandomForestRegressor(random_state=54321, n_estimators= est, max_depth= depth)# inicializa el constructor de modelos con los parámetros random_state=54321, n_estimators=est y max_depth=depth
        model.fit(features_train, target_train) # entrena el modelo en el conjunto de entrenamiento
        predictions_valid = model.predict(features_valid) # obtén las predicciones del modelo en el conjunto de validación
        error = mean_squared_error(target_valid, predictions_valid)**.5# calcula la RECM en el conjunto de validación
        print("Validación RECM para los n_estimators de", est, ", depth=", depth, "is", error)
        if error < best_error: # guardamos la configuración del modelo si se logra el error más bajo
            best_error = error
            best_est = est
            best_depth = depth

print("RECM del mejor modelo en el conjunto de validación:", best_error, "n_estimators:", best_est, "best_depth:", best_depth)
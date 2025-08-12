#Seleccionar el mejor modelo
'''
Hemos entrenado un árbol de decisión, un bosque aleatorio y una regresión lineal. ¿Cuál es el mejor modelo?
A estas alturas, probablemente ya tengas una idea de cuál de los tres modelos de regresión funciona mejor para nuestro ejercicio. 
Si aún no lo tienes claro, repasa las tres lecciones anteriores y compara los valores del RMSE para el conjunto de validación de los modelos que mejor desempeño mostraron. 
Escoge un modelo, configura sus hiperparámetros con valores que demostraron ser óptimos y entrénalo para una prueba final.

Usa todo el dataset fuente para entrenar el modelo que escogiste. Cuantos más datos haya, mejor. 
Ya no necesitamos mantener separado el conjunto de validación porque ya escogimos el modelo más adecuado y ajustamos sus hiperparámetros.
'''

#EJRCICIO
'''
Para aprobar el ejercicio, entrena el mejor modelo con hiperparámetros óptimos en todo el dataset. Si no tienes la seguridad de cuál es el mejor, tómate la libertad de intentarlo con los tres.

Algunos de los modelos que entrenamos lograron obtener valores de RECM de prueba tan bajos como 1.45. Ahora que estamos usando todo el dataset fuente para el entrenamiento podemos esperar que el desempeño del modelo mejore aún más. Intenta romper este récord: a ver si puedes bajar la RECM de prueba a menos de 1.44.

No olvides dividir la variable target entre 100 000 y usar random_state=54321 como en los ejercicios anteriores.
'''

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"C:\Users\josep\Downloads\moved_train_data_us.csv")

features= df.drop(["last_price"], axis=1)
target= df["last_price"]/100000

model = RandomForestRegressor(random_state=54321, n_estimators= 50)
final_model = model.fit(features, target)
predictions = final_model.predict(features)
result= mean_squared_error(target, predictions)**.5
print(result)


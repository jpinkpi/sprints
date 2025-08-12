#Maximización de R2
'''
R2 es más fácil de entender que la RECM. Quedémonos con R2 y escojamos el mejor modelo.
Crea un modelo con el valor R2 más alto posible. Para resolver el ejercicio necesitamos que sea al menos 0.14.

Aquí tienes algunas sugerencias:

Encuentra el R2 del modelo usando la función score(). Esta es la métrica predeterminada para los modelos de regresión en sklearn.
'''

'''
model = LinearRegression()
model.fit(features_train, target_train)
print(model.score(features_valid, target_valid))
'''

'''
Elige la profundidad correcta del árbol. Comencemos con una pequeña cantidad de árboles. 
El número de árboles es proporcional a la calidad del modelo, pero también a la duración del entrenamiento, así que tenlo en mente.

for depth in range(1, 16, 1):
    model = RandomForestRegressor(n_estimators=20, max_depth=depth, random_state=12345)
    model.fit(features_train, target_train)
    # < escribe el código aquí >
'''

#Luego inicia el entrenamiento de bosque aleatorio con una gran cantidad de árboles:
'''
model = RandomForestRegressor(n_estimators=100, 
    max_depth=# < escribe el código aquí>, random_state=12345)
model.fit(features_train, target_train)
print(model.score(features_train, target_train))
print(model.score(features_valid, target_valid))
'''

#Ejercicio
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error

    data = pd.read_csv("/datasets/flights_preprocessed.csv")

    target = data["Arrival Delay"]
    features = data.drop(["Arrival Delay"], axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = RandomForestRegressor(n_estimators = 100, max_depth=11 ) # < escribe el código aquí >
    model.fit(features_train, target_train)

    print("Configuración del modelo actual lograda:")
    print("Valor R2 en un conjunto de entrenamiento", model.score(features_train, target_train))
    print("Valor R2 en un conjunto de validación:", model.score(features_valid, target_valid))
except:print("prueba")

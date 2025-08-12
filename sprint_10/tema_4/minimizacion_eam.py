#Minimización de EAM
'''
Construye un modelo con un valor EAM menor o igual a 26.2.

En la quinta lección de este capítulo ya aprendimos que RandomForestRegressor es una buena alternativa para la regresión lineal. 
Emplea la configuración de modelo que usaste en la lección 5 y verifica a cuál EAM puede llevar.
'''

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error

    data = pd.read_csv("/datasets/flights_preprocessed.csv")

    target = data['Arrival Delay']
    features = data.drop(['Arrival Delay'], axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = RandomForestRegressor(max_depth=11, n_estimators=100, random_state=12345)# < escribe el código aquí >

    model.fit(features_train, target_train)
    predictions_train = model.predict(features_train)
    predictions_valid = model.predict(features_valid)

    print("Configuración del modelo actual lograda:")
    print(
        "Valor EAM en un conjunto de entrenamiento: ",
        mean_absolute_error(target_train, predictions_train),
    )
    print(
        "Valor EAM en un conjunto de validación: ",
        mean_absolute_error(target_valid, predictions_valid),
)
except:print("prueba")
#EMC en una tarea de regresión
'''
¿Qué métrica es la más adecuada para una tarea de regresión? ¡EMC! Entrenemos el modelo y verifiquemos su calidad.
Revisemos cómo calcular ECM (error cuadrático medio) y RECM (raíz cuadrada del error cuadrático medio).

Recuerda que en los ejercicios debéis referiros a los términos en inglés MSE y RMSE, que hacen referencia a ECM y RECM respectivamente.


'''

#EJERCICIO 1
'''
Carga los datos de /datasets/flights_preprocessed.csv. 
Declara la variable predicted_valid. Entrena la regresión lineal. 
Calcula el valor ECM para el conjunto de validación y guárdalo en la variable mse.

Muestra los valores ECM y RECM (en precódigo).
'''
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    data = pd.read_csv('/datasets/flights_preprocessed.csv')

    target = data['Arrival Delay']
    features = data.drop(['Arrival Delay'], axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = LinearRegression()
    model.fit(features_train, target_train)
    predicted_valid = model.predict(features_valid)
    mse = mean_squared_error(target_valid, predicted_valid)
    # < escribe el código aquí >

    print("MSE =", mse)
    print("RMSE =", mse ** 0.5)
except:(print("prueba"))

#EJERCICIO 2
'''

Encuentra los valores de ECM y RECM para el modelo constante: 
esto predice el valor objetivo medio para cada observación. 
Almacena sus predicciones en la variable predicted_valid.

Imprime los valores de ECM y RECM (en precódigo).
'''
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    data = pd.read_csv('/datasets/flights_preprocessed.csv')

    target = data['Arrival Delay']
    features = data.drop(['Arrival Delay'], axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = LinearRegression()
    model.fit(features_train, target_train)
    predicted_valid = model.predict(features_valid)
    mse = mean_squared_error(target_valid, predicted_valid)

    print('Linear Regression')
    print('MSE =', mse)
    print('RMSE =', mse ** 0.5)

    # < escribe el código aquí  >
    predicted_valid = pd.Series(target_train.mean(), index=target_valid.index)
    mse = mean_squared_error(target_valid, predicted_valid)# < escribe el código aquí >)

    print('Mean')
    print('MSE =', mse)
    print('RMSE =', mse ** 0.5)
except:print("prueba")
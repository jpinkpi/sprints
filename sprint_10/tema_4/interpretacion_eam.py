#Interpretación de EAM
'''Cuando calculamos el Error Cuadrático Medio (ECM), a menudo usamos la media de los valores observados como una constante. 
¿Podemos aplicar el mismo principio para el EAM?'''

#EJERCICIO 1
'''
Calcula el EAM utilizando la mediana como valor constante. 
El cálculo EAM para la regresión lineal se encuentra en el precódigo. Compara los valores.

Muestra el resultado en la pantalla.
'''
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error

    data = pd.read_csv('/datasets/flights_preprocessed.csv')

    target = data['Arrival Delay']
    features = data.drop(['Arrival Delay'], axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = LinearRegression()
    model.fit(features_train, target_train)
    predicted_valid = model.predict(features_valid)

    print('Linear Regression')
    print(mean_absolute_error(target_valid, predicted_valid))
    print()

    predicted_valid =pd.Series(target_train.median(), index=target_valid.index) # < escribe el código aquí >
    print('Median')
    print(mean_absolute_error(target_valid,predicted_valid))# < escribe el código aquí >)
except:print("prueba")
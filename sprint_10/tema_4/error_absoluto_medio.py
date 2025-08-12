#Error Absoluto Medio

'''
Vamos a explorar otra métrica de evaluación conocida como EAM (Error Absoluto Medio o “Mean Absolute Error” en Inglés). 
Es similar al ECM (Error Cuadrático Medio), pero no eleva los errores al cuadrado.
'''

#EJERCICIO
'''
1.

Codifica la función mae() según la fórmula. Esta función las respuestas y predicciones correctas y devuelve el valor de error absoluto medio.
Considera que en Python el valor absoluto se calcula usando la función abs().
Prueba la función en el ejemplo en precódigo. Muestra el resultado en la pantalla.

'''
try:
    import pandas as pd

    def mae(target, predictions):
        return (target-predictions).abs().mean()# <  escribe el código aquí >

    target = pd.Series([-0.5, 2.1, 1.5, 0.3])
    predictions = pd.Series([-0.6, 1.7, 1.6, 0.2])

    print(mae(target, predictions))

except:print("prueba")

#EJERCICIO 2
'''
Calcula EAM para la regresión lineal. Encuentra la función apropiada en la documentación de sklearn. Impórtala. Muéstrala en la pantalla.
'''

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    # < escribe el código aquí >

    data = pd.read_csv('/datasets/flights_preprocessed.csv')

    target = data['Arrival Delay']
    features = data.drop(['Arrival Delay'], axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = LinearRegression()
    model.fit(features_train, target_train)
    predicted_valid = model.predict(features_valid)

    print(mean_absolute_error(target_valid, predicted_valid))# < escribe el código aquí >)
except:print("prueba")


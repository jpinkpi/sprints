#Coeficiente de Determinacion


'''
Coeficiente de determinación
Para evitar comparar constantemente el modelo con la media, introduzcamos una nueva métrica. Esta se expresa en valores relativos, no en absolutos.
El coeficiente de determinación o la métrica R2 (R-squared) divide el  ECM del modelo entre el ECM de la media y luego resta el valor obtenido de uno. Si la métrica aumenta, la calidad del modelo también mejora.

R2 se calcula de la siguiente manera:

                        R2 = 1- (Modelo ECM / Media ECM)


R2 es igual a uno solo si el ECM del modelo es cero. Dicho modelo predeciría perfectamente todas las respuestas.
R2 es cero: el modelo funciona tan bien como la media.
Cuando R2 es negativo, peor que usar solo la media para predecir.
R2 no puede tener valores mayores a uno porque esto indicaría que el modelo tiene un poder predictivo negativo, 
lo cual no es posible bajo la definición estándar de esta métrica.                        
'''

#EJERCICIO 

'''
Calcula el valor R2 para la regresión lineal. Busca la función adecuada en la documentación de sklearn.metrics.  Impórtala.

Muestra el resultado (en precódigo).
'''


try: 
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score # < escribe el código aquí >

    data = pd.read_csv('/datasets/flights_preprocessed.csv')

    target = data['Arrival Delay']
    features = data.drop(['Arrival Delay'], axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = LinearRegression()
    model.fit(features_train, target_train)
    predicted_valid = model.predict(features_valid)

    print('R2 =', r2_score(target_valid, predicted_valid))  # < escribe el código aquí >)
except:print("prueba")

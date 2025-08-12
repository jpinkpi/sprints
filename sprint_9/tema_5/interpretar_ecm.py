#Interpretación del ECM
'''
Interpretación del ECM
Para hacer la prueba de cordura al modelo para buscar problemas de clasificación necesitamos comparar sus predicciones con posibilidad aleatoria. 
¿Cómo se hace una prueba de cordura para un modelo de regresión?

Darle la misma respuesta a todas las observaciones es un método simple de predicción de regresión. 
Vamos a usar el valor promedio del precio del apartamento para estar más cerca de la realidad.
'''

#EJERCICIO 1
'''
Prepara los datos y encuentra el precio promedio.

Declara la variable features con todas las características excepto last_price.
Declara la variable target con last_price como objetivo.
Calcula el valor promedio para los elementos de la variable target.
Imprime el resultado como se muestra:
'''

try:
    import pandas as pd
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv('/datasets/train_data_us.csv')
    features = df.drop(["last_price"], axis=1)
    target = df["last_price"]

    print("Average price:", target.mean())
    # < crea las variables features y target >

    # < calcula e imprime el promedio >
except: print("Prueba")

#EJERCICIO 2
'''
Calcula el ECM para el conjunto de entrenamiento usando el precio promedio como valor de predicción.

Imprime el resultado como se muestra:

MSE: ...
La función mean_squared_error() en scikit-learn tiene su maña. 
Tendrás que pasar unos minutos leyendo la documentación o consultar Stack Overflow.

En el precódigo, dividimos los precios entre 100 000 con el propósito de evitar números grandes.
'''

try:
    import pandas as pd
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv('/datasets/train_data_us.csv')

    features = df.drop(['last_price'], axis=1)
    target = df['last_price'] / 100000
    predictions = pd.Series(target.mean(), index= target.index)
    mse = mean_squared_error(target, predictions)
    # < calcula el ECM  >

    print('MSE:', mse)
except:print("prueba")

#EJERCICIO 3
'''
No necesitamos "dólares cuadrados". Para obtener dólares normales encuentra la RECM sacando la raíz cuadrada del ECM. Imprime en la pantalla los resultados como se muestra: 

RMSE: ...
Para encontrar la raíz cuadrada de un número usa el operador de exponente **. Eleva el número a la potencia 0.5. Por ejemplo: 

print(25 ** 0.5)
5
'''

try:
    import pandas as pd
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv('/datasets/train_data_us.csv')

    features = df.drop(['last_price'], axis=1)
    target = df['last_price'] / 100000

    predictions = pd.Series(target.mean(), index=target.index)
    mse = mean_squared_error(target, predictions)
    rmse = mse ** .5
    # < encuentra la raíz cuadrada de ECM  >

    print('RMSE:', rmse)

except: print("prueba")
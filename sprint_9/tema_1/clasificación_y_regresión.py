# Clasificación y regresión
"""""
Volvamos a nuestro problema de vivienda y decidamos qué es mejor usar: clasificación o regresión.

El precio del apartamento es un objetivo numérico, por lo que se trata de una tarea de regresión. 
La regresión suele implicar largos cálculos con muchas respuestas posibles, por lo que las tareas 
de regresión no son la forma más sencilla de familiarizarse con el machine learning. 
Para simplificar, dividiremos todos los precios en "altos" y "bajos" por ahora, 
convirtiendo efectivamente nuestra tarea en una tarea de clasificación binaria con solo dos respuestas posibles. 
Entonces todo lo que tenemos que hacer es predecir en qué clase cae cualquier lista dada. Nos ocuparemos de la regresión más tarde.

Entonces, ¿cómo se dividen exactamente los precios en altos y bajos? Es más fácil cuando hay un número casi igual de objetos en las categorías. 
¿Por qué? ¡Porque es difícil distinguir las cornejas de los cuervos si solo estás viendo a las cornejas!

Averigüemos el precio medio (justo en el medio).

""" 

try:
    import pandas as pd

    df = pd.read_csv('/datasets/train_data_us.csv')

    print(df['last_price'].median())

except:
    print("df de prueba")

#EJERCICIO 1 
try:
    import pandas as pd
    import numpy as np

    df = pd.read_csv('/datasets/train_data_us.csv')
    df.loc[df["last_price"]> 113000, "price_class"] = 1
    df.loc[df["last_price"] <= 113000, "price_class"] = 0
        
    # < escribe el código aquí >

    print(df.head())
except:
    print("df de prueba")

#EJERCICIO 2
try:
    import pandas as pd

    df = pd.read_csv('/datasets/train_data_us.csv')

    df.loc[df['last_price'] > 113000, 'price_class'] = 1
    df.loc[df['last_price'] <= 113000, 'price_class'] = 0

    features = df.drop(["last_price", "price_class"], axis=1)# < escribe el código aquí >
    target = df["price_class"] # < escribe el código aquí >

    print(features.shape)
    print(target.shape)

except:
    print("df de prueba")
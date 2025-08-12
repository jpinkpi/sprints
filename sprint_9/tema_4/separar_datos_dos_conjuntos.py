#Separar datos en dos conjuntos
'''
El conjunto de validación constituye el 25 % de los datos fuente. Entonces, ¿cómo debemos extraerlo?

En sklearn hay una función llamada train_test_split, con la que se puede separar cualquier conjunto de datos en dos: entrenamiento y prueba. Pero nosotros vamos a usar esta función para obtener un conjunto de validación y uno de entrenamiento.

Importa train_test_split desde el módulo model_selection de scikit-learn:

from sklearn.model_selection import train_test_split
Antes de separar, necesitamos establecer dos parámetros:

Nombre del dataset que vamos a separar.
Tamaño del conjunto de validación (test_size). El tamaño se expresa con un decimal entre 0 y 1 que representa una fracción del dataset fuente. En este caso, tenemos test_size=0.25 porque queremos trabajar con el 25 % del conjunto fuente.
La función train_test_split() devuelve dos conjuntos de datos: entrenamiento y validación.

df_train, df_valid = train_test_split(df, test_size=0.25, random_state=12345)
Nota: podemos asignar cualquier valor a random_state excepto None.
'''

#EJERCICIO 1
'''
Separa el dataset en dos conjuntos:

conjunto de entrenamiento (df_train);
conjunto de validación (df_valid) — 25 % de los datos fuente
Declara cuatro variables y pásalas de la siguiente manera:

características: features_train y features_valid;
objetivo: target_train y target_valid
Imprime los tamaños de las tablas que están almacenadas en cuatro variables (hecho en el precódigo).
'''

try: 
    import pandas as pd
    from sklearn.model_selection import train_test_split
    # < importa la función train_test_split desde la librería sklearn >

    df = pd.read_csv('/datasets/train_data_us.csv')
    df.loc[df['last_price'] > 113000, 'price_class'] = 1
    df.loc[df['last_price'] <= 113000, 'price_class'] = 0
    df_train, df_valid = train_test_split(df, test_size= 0.25, random_state= 12345)
    # < separa los datos en entrenamiento y validación >

    # < declara variables para las características y para la característica objetivo >
    features_train = df_train.drop(['last_price', 'price_class'], axis=1)
    target_train = df_train['price_class']
    features_valid = df_valid.drop(['last_price', 'price_class'], axis=1)# < escribe tu código aquí>
    target_valid = df_valid['price_class'] # < escribe el código aquí

    print(features_train.shape)
    print(target_train.shape)
    print(features_valid.shape)
    print(target_valid.shape)
except:print("prueba")

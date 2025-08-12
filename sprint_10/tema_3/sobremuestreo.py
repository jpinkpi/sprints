#Sobremuestreo
'''
¿Cómo podemos hacer que las observaciones de una clase rara sean menos raras en los datos?

Ahora obtienes 1 punto por resolver cualquier tarea en la prueba. Las tareas más importantes se repiten varias veces para que sean más fáciles de recordar.

En el entrenamiento de modelos, esta técnica se denomina sobremuestreo.

El sobremuestreo se realiza, por lo general, en cuatro pasos:

Se divide el dataset de entrenamiento en observaciones negativas y positivas.
Se duplican varias veces las observaciones positivas (las que raramente ocurren).
Se crea una nueva muestra de entrenamiento con base en los datos obtenidos.
Se mezclan los datos: hacer la misma pregunta una y otra vez no ayudará al entrenamiento.
Comencemos por implementar estos pasos en los siguientes ejercicios. Te guiaremos a lo largo de todo el proceso.

'''

#EJERCICIO 1
'''
Hemos dividido los datos en datos de entrenamiento y datos de prueba. 
Tu tarea consistirá en dividir el dataset de entrenamiento en cuatro variables del siguiente modo:

features_zeros — características de las observaciones de la respuesta "0"
features_ones — características de las observaciones de la respuesta "1"
target_zeros — objetivo de las observaciones de la respuesta "0"
target_ones — objetivo de las observaciones de la respuesta "1"

Muestra en la pantalla los tamaños de las cuatro tablas almacenadas en las variables (ya en el precódigo).

'''

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    # < escribe el código aquí>
    features_zeros= features_train[target_train == 0]
    features_ones= features_train[target_train == 1]
    target_zeros=target_train[target_train ==0]
    target_ones =target_train[target_train ==1]


    print(features_zeros.shape)
    print(features_ones.shape)
    print(target_zeros.shape)
    print(target_ones.shape)
except:print("prueba")

#EJERCICIO 2
'''

Se pueden duplicar observaciones "n" número de veces utilizando la sintaxis de multiplicación de listas de Python. 
Para repetir los elementos de la lista, la lista se multiplica por un número entero (el número requerido de repeticiones). 
Aquí tienes un ejemplo:

'''
answers = [0, 1, 0]
print(answers)
answers_x3 = answers * 3
print(answers_x3)
#RESULTADO
[0, 1, 0]
[0, 1, 0, 0, 1, 0, 0, 1, 0]


'''
Tu tarea es abordar el desequilibrio de clases al: 

1.Identicar la clase subrepresentada (clase positiva).
2.Duplicar estas observaciones de clase positivas. El número de repeticiones se almacena en la variable repeat.
3.Combinar las observaciones de clase positivas duplicadas con las observaciones de clase negativas.
4.Utiliza la función pd.concat() para realizar esta combinación.
5.Consultar la documentación de pd.concat() para conocer los detalles sobre cómo utilizar la función.
6.Almacenar las características combinadas en la variable features_upsampled.
7.Repetir el proceso de duplicación para el objetivo y almacena el resultado en la variable target_upsampled.
8.Muestra los tamaños de las nuevas variables (en el precódigo).
'''

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    features_zeros = features_train[target_train == 0]
    features_ones = features_train[target_train == 1]
    target_zeros = target_train[target_train == 0]
    target_ones = target_train[target_train == 1]

    repeat = 10
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled =pd.concat([target_zeros] + [target_ones]*repeat) # <escribe el código aquí  >

    print(features_upsampled.shape)
    print(target_upsampled.shape)
except:print("prueba")


#EJERCICIO 3

'''

Utilizaremos lo que hemos programado en los ejercicios anteriores y lo implementaremos dentro de la función upsample(). La función tiene que:

1.Manejar tres parámetros: features, target y repeat .
2.Dividir features y target en dataframes/series para la clase 0 y la clase 1. (Ejercicio 1)
3.Duplicar: Repetir los datos de la clase 1 repeat veces (Ejercicio 2).
4.Concatenar: Combinar los datos de la clase 0 con los datos repetidos de la clase 1 utilizando pd.concat() (Ejercicio 2).
5.Mezclar los datos combinados con shuffle() con un random_state de tu elección.
6.Devolver las características mezcladas y el objetivo.

Por último, llamarás a la función de los datos entrenados y mostrarás los tamaños de las muestras (en el precódigo).
'''
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    # < crear la función a partir del siguiente código >
    def upsample(features, target, repeat):
        features_zeros = features[target == 0]
        features_ones = features[target == 1]
        target_zeros = target[target == 0]
        target_ones = target[target == 1]

        repeat = 10
        features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
        target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
        features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=12345)
    # < añadir aleatorio >
        return features_upsampled, target_upsampled

    features_upsampled, target_upsampled = upsample(
            features_train, target_train, 10
    )

    print(features_upsampled.shape)
    print(target_upsampled.shape)
except:print("prueba")

#Ejercicio 4
'''
Entrena el modelo LogisticRegression con los nuevos datos. 
Encuentra el valor F1 y muéstralo en la pantalla (en precódigo).
'''
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    def upsample(features, target, repeat):
        features_zeros = features[target == 0]
        features_ones = features[target == 1]
        target_zeros = target[target == 0]
        target_ones = target[target == 1]

        features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
        target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

        features_upsampled, target_upsampled = shuffle(
            features_upsampled, target_upsampled, random_state=12345
        )

        return features_upsampled, target_upsampled


    features_upsampled, target_upsampled = upsample(
        features_train, target_train, 10
    )
    model = LogisticRegression(random_state=12345,solver='liblinear')
    model.fit(features_upsampled, target_upsampled)
    predicted_valid = model.predict(features_valid)
    # < escribe el código aquí >

    print('F1:', f1_score(target_valid, predicted_valid))
except:print("prueba")


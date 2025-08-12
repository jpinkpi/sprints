#Submuestreo

'''
El submuestreo es una técnica que se utiliza para equilibrar las clases dentro de un conjunto de datos de entrenamiento.
Al reducir la frecuencia de las observaciones de una clase predominante, 
podemos mejorar el rendimiento del modelo de aprendizaje automático al evitar que esté sesgado hacia la clase más común. 
¿Cómo podemos hacer que las observaciones de una clase frecuente sean menos frecuentes en los datos?
En lugar de repetir las preguntas importantes, también podemos eliminar una parte de las que no son importantes. 
Para ello, podemos utilizar la técnica de submuestreo.

El submuestreo se realiza en varios pasos:

División del conjunto de datos: separar el dataset de entrenamiento en dos grupos, uno para cada clase (positiva y negativa).

Selección aleatoria: 
usa la función sample() para eliminar aleatoriamente una proporción de las observaciones de la clase predominante (negativa en este caso).

Creación de una nueva muestra de entrenamiento: 
combina las observaciones restantes de la clase negativa con todas las observaciones de la clase positiva para forma un nuevo conjunto de entrenamiento.

Mezcla de datos: 
si los datos no están bien mezclados, el modelo podría aprender patrones que no son realmente representativos del problema que se está abordando, 
sino que simplemente reflejan el orden en que los datos se presentaron durante el entrenamiento.


Para eliminar aleatoriamente algunos elementos de la tabla, utiliza la función sample(). 
Esta función requiere un parámetro llamado frac ('fraction' o fracción), que especifica la proporción de los elementos totales que quieres retener. 
Por ejemplo, frac=0.1 significa que la función seleccionará aleatoriamente el 10% de los elementos de la tabla original para formar una nueva muestra.
'''

# print(features_train.shape)

# features_sample = features_train.sample(frac=0.1, random_state=12345)
# print(features_sample.shape)

'Especifica random_state=12345 para que tu código sea más fácil de revisar.'


#EJERCIO
'''

Para realizar el submuestreo, escribe una función downsample() y pásale tres argumentos:

-features
-target
-fraction (fracción de observaciones negativas a mantener)

La función debe realizar el submuestreo y devolver tanto las características modificadas como las etiquetas. 
Antes de devolver estos datos, asegúrate de mezclarlos para evitar sesgos en el proceso de aprendizaje del modelo.

Llama a la función para los datos de entrenamiento con una fracción de 0.1. Muestra en la pantalla los tamaños de las muestras (en precódigo).
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

    def downsample(features, target, fraction):
        features_zeros = features[target == 0]
        features_ones = features[target == 1]
        target_zeros = target[target == 0]
        target_ones = target[target == 1]

        features_downsampled= pd.concat([features_zeros.sample(frac=fraction, random_state=12345)]+ [features_ones])
        target_downsampled = pd.concat([target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])
        features_downsampled,target_downsampled = shuffle(features_downsampled, target_downsampled, random_state=12345)# < escribe el código aquí >

        return features_downsampled, target_downsampled


    features_downsampled, target_downsampled = downsample(
        features_train, target_train, 0.1
    )

    print(features_downsampled.shape)
    print(target_downsampled.shape)
except:print("prueba")


#EJERCICIO 2
'Entrena el modelo LogisticRegression con los nuevos datos. Encuentra el valor F1 y muéstralo en la pantalla (en precódigo).'


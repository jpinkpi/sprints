#Algoritmo de k vecinos más cercanos (K-Nearest Neighbors)
'''
Ahora que hemos estudiado las dos métricas de distancia más comunes, vamos a ponerlas en práctica a través de un algoritmo de machine learning real.
El algoritmo K-Nearest Neighbors (k-NN) es un algoritmo popular de machine learning basado en la distancia que puede ser utilizado a la hora de realizar tareas de clasificación y regresión.

Observa la imagen de abajo. Si queremos clasificar una nueva observación (representada por el signo de interrogación) 
en una de las dos clases (azul o roja), basta con encontrar la observación más cercana y asignar su clase a nuestro nuevo punto. 
Al comparar dos observaciones, podemos utilizar tanto la distancia euclidiana como la Manhattan.

El algoritmo k-NN funciona de acuerdo con el mismo principio.

En el ejemplo anterior, hemos utilizado solo la observación más cercana para clasificar nuestra nueva observación. 
Sin embargo, el algoritmo nos da la posibilidad de definir el número de vecinos de interés en este proceso de clasificación.


Así es como funciona el algoritmo k-NN:

                                Paso 1: Seleccionamos el número (
                                k
                                k) de vecinos de referencia.
                                Paso 2: Calculamos la distancia entre la nueva observación y los demás puntos de nuestros datos. La distancia euclidiana se selecciona por defecto, pero también podemos utilizar las distancias Manhattan, Minkowski o Hamming.
                                Paso 3: Ordenamos las distancias de la más cercana a la más lejana y consideramos los vecinos 
                                k
                                k más cercanos.
                                Paso 4: Finalmente, clasificamos nuestra nueva observación en la misma clase a la que pertenece la mayoría de los vecinos.

El algoritmo funciona tanto en un plano como en un espacio multidimensional.

Podemos importar el algoritmo k-NN de la librería sklearn con el fin de resolver problemas. 
Sin embargo, lo creas o no, ya hemos aprendido lo suficiente como para construir este algoritmo desde cero. 
Así que, vamos a intentarlo.

Volvamos a esa agencia inmobiliaria, Cribswithclass.com. 
Un usuario ha añadido una nueva propiedad pero ha olvidado indicar en el listado que hay 2 dormitorios. 
Vamos a ver si podemos utilizar nuestra base de datos para adivinar el número de habitaciones y rellenar ese campo por él de forma automática.
'''


#EJERCICIO
'''
Escribe la función nearest_neighbor_predict(). Esta función toma tres argumentos:

características del conjunto de entrenamiento (train_features)
objetivo del conjunto de entrenamiento (train_target)
nuevas características de observación (new_features)

La función utiliza el algoritmo de vecinos más cercanos para devolver una predicción objetivo para la nueva observación.
Ejecuta la función para la nueva observación (new_apartament). 
Comprueba si podemos predecir correctamente el número de dormitorios. Muestra los resultados (en precódigo).
'''





import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

columns = [
    'bedrooms',
    'total area',
    'kitchen',
    'living area',
    'floor',
    'total floors',
]

df_train = pd.DataFrame(
    [
        [1.0, 38.5, 6.9, 18.9, 3.0, 5.0],
        [1.0, 38.0, 8.5, 19.2, 9.0, 17.0],
        [1.0, 34.7, 10.3, 19.8, 1.0, 9.0],
        [1.0, 45.9, 11.1, 17.5, 11.0, 23.0],
        [1.0, 42.4, 10.0, 19.9, 6.0, 14.0],
        [1.0, 46.0, 10.2, 20.5, 3.0, 12.0],
        [2.0, 77.7, 13.2, 39.3, 3.0, 17.0],
        [2.0, 69.8, 11.1, 31.4, 12.0, 23.0],
        [2.0, 78.2, 19.4, 33.2, 4.0, 9.0],
        [2.0, 55.5, 7.8, 29.6, 1.0, 25.0],
        [2.0, 74.3, 16.0, 34.2, 14.0, 17.0],
        [2.0, 78.3, 12.3, 42.6, 23.0, 23.0],
        [2.0, 74.0, 18.1, 49.0, 8.0, 9.0],
        [2.0, 71.4, 20.1, 60.4, 2.0, 10.0],
        [3.0, 85.0, 17.8, 56.1, 14.0, 14.0],
        [3.0, 79.8, 9.8, 44.8, 9.0, 10.0],
        [3.0, 72.0, 10.2, 37.3, 7.0, 9.0],
        [3.0, 95.3, 11.0, 51.5, 15.0, 23.0],
        [3.0, 69.3, 9.5, 42.3, 4.0, 9.0],
        [3.0, 89.8, 11.2, 58.2, 24.0, 25.0],
    ],
    columns=columns,
)


def nearest_neighbor_predict(train_features, train_target, new_features):
    distances = []
    for i in range(len(train_features)):
        d = distance.euclidean(train_features.iloc[i], new_features)
        distances.append(d)
    nearest_index = np.argmin(distances) #Equivalente a k = 1 
    return train_target.iloc[nearest_index]
    # < escribe tu código aquí >
    

train_features = df_train.drop('bedrooms', axis=1)
train_target = df_train['bedrooms']
new_apartment = np.array([72, 14, 39, 8, 16])
prediction = nearest_neighbor_predict(
    train_features, train_target, new_apartment
)
print(prediction)


#En caso de querer interpretarlo(CHATGPT xd)
# Preparamos datos sin la columna objetivo
features = df_train.drop('bedrooms', axis=1)

# Aplicamos PCA para reducir a 2 dimensiones
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

# Graficamos, coloreando por número de recámaras
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    features_pca[:, 0], features_pca[:, 1],
    c=df_train['bedrooms'], cmap='viridis', s=100, edgecolors='k'
)
plt.title('Apartamentos proyectados en 2D con PCA')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.colorbar(scatter, label='Número de recámaras')
plt.grid(True)
plt.show()
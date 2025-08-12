#Validación cruzada en Python

'''
Completa el código del bucle para obtener una validación cruzada en tres bloques del mismo tamaño. 
En cada etapa del ciclo, tienes el número del primer elemento de la muestra de validación y el tamaño del bloque (sample_size).

Crea las matrices: valid_indexes y train_indexes. Estas contienen números de observaciones para muestras de validación y entrenamiento. 
Cambia los números en cada etapa del ciclo.

Divide las variables features y target en las muestras features_train, target_train, features_valid y 
target_valid para que contengan solo observaciones con los números necesarios.

Evalúa la calidad del modelo entrenado en cada muestra.
Calcula la calidad promedio del modelo y guárdala en la variable final_score. Imprime el valor de la pantalla (en precódigo).

'''

try:
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    data = pd.read_csv('/datasets/heart.csv')
    features = data.drop(['target'], axis=1)
    target = data['target']

    scores = []

    # establece el tamaño del bloque si solo hay tres de ellos
    sample_size = int(len(data)/3)

    for i in range(0, len(data), sample_size):
        valid_indexes = list(range(i,i + sample_size))# < escribe una matriz de índices para el bloque de validación >
        train_indexes = list(range(0,i)) + list(range(i+sample_size, len(data))) # < escribe una matriz de índices para el conjunto de entrenamiento >
        features_train = features.iloc[train_indexes]
        features_valid = features.iloc[valid_indexes]

        target_train = target.iloc[train_indexes]
        target_valid = target.iloc[valid_indexes]
            # Divide las características de las variables y el objetivo en muestras features_train, target_train, features_valid, target_valid
        # < escribe tu código aquí >

        model = DecisionTreeClassifier(random_state=0)
        model = model.fit(features_train, target_train)
        score = model.score(features_valid, target_valid) # < evalúa la calidad del modelo >
        
        scores.append(score)
    
    # < calcula la calidad media del modelo >  
    final_score = sum(scores) /len(scores)
    print('Valor de calidad promedio del modelo:', final_score)
except: print("prueba")


#Tambien de manera mas fácil, se puede realizar esto en sklearn
'''
Para evaluar el modelo por validación cruzada usaremos la función cross_val_score del módulo sklearn.model_selection.

Así se llama a la función:

from sklearn.model_selection import cross_val_score

cross_val_score(model, features, target, cv=3)
La función toma varios argumentos, como:

model: modelo para validación cruzada. Será entrenado en el proceso de validación cruzada, por lo que tenemos que pasarlo sin entrenar. Supongamos que necesitamos este modelo para un árbol de decisión:
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
features
target
cv — número de bloques para validación cruzada (son 3, por defecto)
La función no requiere dividir los datos en bloques o muestras para la validación y el entrenamiento. Todos estos pasos se realizan de forma automática. La función devuelve una lista de valores de evaluación del modelo de cada validación. Cada valor es igual a model.score() para la muestra de validación. Por ejemplo, para una tarea de clasificación, esto es exactitud.
'''


#EJERCICIO

try:
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score

    data = pd.read_csv('/datasets/heart.csv')
    features = data.drop(['target'], axis=1)
    target = data['target']

    model = DecisionTreeClassifier(random_state=0)
    cvs = cross_val_score(model, features, target, cv=5)
    final_score = cvs.mean()
    # < calcula las puntuaciones llamando a la función cross_val_score con cinco bloques >

    print('Puntuación media de la evaluación del modelo:', final_score)
except:print("prueba")
#Matriz de confusión

'''
Cuando VP, FP, VN, FN se recopilan en una tabla, se denomina matriz de confusión.
La matriz se forma de la siguiente manera:

las etiquetas del algoritmo (0 y 1) se colocan en el eje horizontal ("Predicciones");
las etiquetas verdaderas de la clase (0 y 1) se colocan en el eje vertical ("Respuestas").
Lo que obtienes:

Las predicciones correctas están en la diagonal principal (desde la esquina superior izquierda):
VN en la esquina superior izquierda
VP en la esquina inferior derecha
Las predicciones incorrectas están fuera de la diagonal principal:
 
FP en la esquina superior derecha
FN en la esquina inferior izquierda

La matriz de confusión te permite visualizar los resultados de calcular las métricas de exactitud y recall.

La matriz de confusión se encuentra en el módulo sklearn.metrics, que ya conoces. La función confusion_matrix() toma respuestas y predicciones correctas y devuelve una matriz de confusión.

'''


#EJERCICIO 1
'''
Calcula la matriz de confusión utilizando la función confusion_matrix().  Impórtala desde el módulo sklearn.metrics. Muestra los resultados en la pantalla.
'''
import pandas as pd
from sklearn.metrics import confusion_matrix
# < escribe el código aquí  >

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

matriz = confusion_matrix(target, predictions)
print(matriz)
# < escribe el cód

#EJERCICIO 2
'''
Calcula una matriz de confusión para el árbol de decisión y llama a la función confusion_matrix(). Muéstrala en pantalla. Muéstrala en pantalla.
'''
try: 
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = DecisionTreeClassifier(random_state=12345)
    model.fit(features_train, target_train)
    predicted_valid = model.predict(features_valid)
    matriz = confusion_matrix(target_valid, predicted_valid)
    print(matriz)
except:print("prueba")


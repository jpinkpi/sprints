#Proyecto sprint 9

'''
Como parte del proyecto tenemos como obejtivo el análisis de los clientes para la compañía de telefonía móvil Megaline, 
me enfrenté al reto de desarrollar un modelo de clasificación que permita predecir el plan más adecuado para cada usuario. 
Actualmente, la empresa ha detectado que muchos de sus clientes siguen utilizando planes antiguos, 
lo cual representa un obstáculo para la adopción de sus nuevos productos: los planes Smart y Ultra. 
Para abordar esta situación, trabajé con un conjunto de datos que contiene el comportamiento de los usuarios que ya se han cambiado a los nuevos planes.
Dado que el preprocesamiento de los datos ya se ha realizado en un proyecto anterior (durante el sprint de Análisis Estadístico de Datos), 
me enfoqué directamente en la creación del modelo predictivo. 
El objetivo fue construir un modelo que clasifique correctamente a los usuarios según el plan que más les conviene, con una exactitud mínima de 0.75, 
usando métricas de validación para comprobar su rendimiento.
'''

#Abre y examina el archivo de datos. Dirección al archivo:/datasets/users_behavior.csv Descarga el dataset
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r"C:\Users\josep\Downloads\users_behavior.csv")
df.info()
df["calls"] = df["calls"].astype("int")
df["messages"] = df["messages"].astype("int")
print(df.sample(n=5))

'''
A simple vista no detecto ningún error; sin embargo, convertiré los valores de las columnas “calls” y “messages” de tipo flotante a enteros, 
ya que representan conteos discretos.
'''


#Segmenta los datos fuente en un conjunto de entrenamiento, uno de validación y uno de prueba.

df_train, df_valid = train_test_split(df, random_state=54321, train_size=.60)

df_valid, df_test = train_test_split(df_valid, random_state=54321, test_size=.50)

print(df_train.shape)
print(df_valid.shape)
print(df_test.shape)

'''
Con base en lo aprendido en capítulos anteriores, segmenté el conjunto de datos utilizando la función train_test_split. 
Asigné el 60% de los datos al conjunto de entrenamiento y reservé el 40% restante. Posteriormente, dividí ese 40% en partes iguales: 
20% para el conjunto de validación y 20% para el conjunto de prueba. De esta manera, los datos quedaron distribuidos en un 60% para entrenamiento, 
20% para validación y 20% para prueba. 
Por ultimo muestro el tamaño de los conjuntos para confirmar lo hecho.
'''


#Investiga la calidad de diferentes modelos cambiando los hiperparámetros. Describe brevemente los hallazgos del estudio

#Arbol de desicion
features_train = df_train.drop(["is_ultra"],axis= 1)
target_train = df_train["is_ultra"]
features_valid = df_valid.drop(["is_ultra"],axis= 1)
target_valid = df_valid["is_ultra"]
features_test= df_test.drop(["is_ultra"],axis= 1)
target_test = df_test["is_ultra"]

best_score = 0
best_depth = 0

for depth in range(1,20):
        model= DecisionTreeClassifier(random_state=12345, max_depth=depth)
        model.fit(features_train, target_train)
        predictions_valid = model.predict(features_valid)
        score = accuracy_score(target_valid, predictions_valid)

        if score > best_score:
                    best_score = score 
                    best_depth = depth

print("La exactitud del mejor modelo utlizando un arbol de desicionen en el conjunto de validación (max_depth = {}): {}".format(best_depth, best_score))

'''
Los resultados obtenidos al utilizar el árbol de decisión, aplicando un bucle para determinar la profundidad óptima del modelo, 
indican que la mejor precisión se alcanza con una profundidad de 10. Este modelo logró una puntuación de 0.7791, 
superando el umbral mínimo requerido de 0.75. 
Sin embargo, continuaremos explorando otras alternativas con el objetivo de encontrar un modelo con un rendimiento aún mejor.
'''

#bosque aleatorio

best_score = 0
best_est = 0
for est in range(1, 51): 
    model = RandomForestClassifier(random_state=54321, n_estimators= est) 
    model.fit(features_train,target_train) 
    score = model.score(features_valid,target_valid) 
    if score > best_score:
        best_score = score 
        best_est = est 

print("La exactitud del mejor modelo utlizando un bosque aleatorio en el conjunto de validación (n_estimators = {}): {}".format(best_est, best_score))
'''
El mejor modelo obtenido utilizando un bosque aleatorio con 44 estimadores alcanzó una exactitud de aproximadamente 0.7854 en el conjunto de validación.
Este resultado supera nuestro umbral de desempeño esperado, lo que indica que el modelo es efectivo para la tarea de clasificación. 
Sin embargo, seguiremos evaluando otras opciones para optimizar aún más los resultados.
'''

#Regresion logistica

model =  LogisticRegression(random_state=54321, solver="liblinear")
model.fit(features_train, target_train) 
score_train = model.score(features_train, target_train) 
score_valid = model.score(features_valid, target_valid) 

print("Accuracy del modelo de regresión logística en el conjunto de entrenamiento:", score_train)
print("Accuracy del modelo de regresión logística en el conjunto de validación:", score_valid)

'''
El modelo de regresión logística alcanzó una exactitud de aproximadamente 0.7132 en el conjunto de entrenamiento y 0.6781 en el conjunto de validación.
Aunque el rendimiento es moderado y menor que otros modelos evaluados, decidimos finalizar con este modelo, concluyendo que, dadas sus limitaciones, 
no supera las alternativas previas para esta tarea de clasificación.

Por los resultados obtenidos a lo largo de las pruebas, decidimos quedarnos con la práctica del bosque aleatorio, 
ya que fue el modelo que alcanzó la mayor exactitud en el conjunto de validación, 
con un valor de 0.7854, superando con claridad el umbral establecido
'''


#Comprueba la calidad del modelo usando el conjunto de prueba.(bosque aleatorio)

model = RandomForestClassifier(random_state=54321, n_estimators= 44) 
model.fit(features_train,target_train)
score = model.score(features_test,target_test)
print("La exactitud del bosque aleatorio en el conjunto de prueba utilizando (n_estimators = {}): {}".format(best_est, score))
'''
Finalmente, al evaluar el modelo de bosque aleatorio con 44 estimadores en el conjunto de prueba, 
obtuvimos una exactitud de aproximadamente 0.8196. Este resultado confirma que el modelo funciona bien con datos nuevos y
respalda nuestra decisión de utilizarlo como la mejor opción para esta tarea de clasificación
'''


#Tarea adicional: haz una prueba de cordura al modelo. 
#Estos datos son más complejos que los que habías usado antes así que no será una tarea fácil.
#Más adelante lo veremos con más detalle.











#Nuevos modelos: bosque aleatorio
'''
Hemos cambiado los hiperparámetros pero los resultados aún dejan mucho que desear. Claramente, un árbol no es suficiente. ¡Necesitamos un bosque!
Probemos con un nuevo algoritmo de aprendizaje llamado bosque aleatorio. Este algoritmo entrena una gran cantidad de árboles independientes y toma una decisión mediante el voto.
 Un bosque aleatorio ayuda a mejorar los resultados y a evitar el sobreajuste.

Sabes por qué la gente vota cuando hay que tomar decisiones importantes, ¿verdad? De esta forma puedes obtener una valoración promedio que anule el sesgo personal y los errores. 
El bosque aleatorio se basa en el mismo principio.
Entonces, ¿cómo lo entrenamos? En la librería scikit-learn, puedes encontrar RandomForestClassifier que es un algoritmo de bosque aleatorio. 
'''
#Impórtalo desde el módulo ensemble:



from sklearn.ensemble import RandomForestClassifier
'''
Usaremos el hiperparámetro n_estimators (significa "número de estimadores") para establecer el número de árboles en el bosque. 
El aumento en la cantidad de estimadores siempre disminuye la varianza de la predicción, por lo que cuantos más árboles uses, mejores resultados obtendrás. 
Los bosques no pueden sobreajustarse debido a que tienen demasiados árboles. Si bien el sobreajuste de un bosque aún puede ocurrir debido al sobreajuste de sus árboles individuales, 
este efecto generalmente se ve compensado por el beneficio de tener muchos árboles. En casos raros donde no lo es, la poda lo arregla, pero en la mayoría de los casos los beneficios de la poda son insignificantes.

Aunque el número de estimadores nunca provoca un sobreajuste, sigue siendo necesario limitarlo, aunque por una razón diferente. 
El uso de más y más árboles incurre en un costo computacional cada vez mayor y sufre de rendimientos decrecientes. Eventualmente, 
la métrica de calidad del modelo alcanza una meseta y deja de mejorar, mientras que el tiempo de ejecución sigue aumentando.
'''

#Scikit-learn establece n_estimators en 100 de forma predeterminada. Pero por ahora, vamos a establecer el valor de n_estimators en 3. 
#Y no olvides hacer que la pseudoaleatoriedad sea estática con el parámetro random_state.

model = RandomForestClassifier(random_state=54321, n_estimators=3)

#Como en las lecciones anteriores, vamos a entrenar el modelo usando el método fit().

#model.fit(features, target)
#Hasta este punto usamos la función accuracy_score() para comparar las etiquetas predichas con las respuestas reales y cuantificar las discordancias. 
# Sin embargo, si todo lo que queremos hacer es evaluar la calidad del modelo, y no nos importan las etiquetas predichas en sí mismas, en lugar de usar el método predict() con la función accuracy_score(), 
# podemos usar un método que llama a ambos de manera interna: el método score(). De esta manera, el paso intermedio de convertir características en predicciones está oculto para nosotros y, en cambio, obtenemos la puntuación de exactitud de inmediato. 
# Usarlo hace que el código sea más claro y más corto. Así es como se le llama:

#model.score(features, target)

#EJERCICIO 1
'''
Divide los datos en conjuntos de entrenamiento y validación.
Elige el rango para que sea lo suficientemente grande para obtener una puntuación lo suficientemente buena, pero lo suficientemente pequeño para que tu programa no sea innecesariamente lento.
Configura el número de árboles para que sea igual a la variable de bucle est en el constructor del modelo.
Entrena modelos en el conjunto de entrenamiento. Calcula accuracy en el conjunto de validación para cada modelo.
Imprime la mejor puntuación de accuracy junto con el número correspondiente de estimadores.
'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\josep\Downloads\moved_train_data_us.csv")

df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0

df_train, df_valid = train_test_split(df, test_size=0.25, random_state=54321) # segmenta el 25% de los datos para hacer el conjunto de validación
features_train = df_train.drop(['last_price', 'price_class'], axis=1)
target_train = df_train['price_class']
features_valid = df_valid.drop(['last_price', 'price_class'], axis=1)
target_valid = df_valid['price_class']

best_score = 0
best_est = 0
for est in range(1, 11): # selecciona el rango del hiperparámetro
    model = RandomForestClassifier(random_state=54321, n_estimators= est) # configura el número de árboles
    model.fit(features_train,target_train) # entrena el modelo en el conjunto de entrenamiento
    score = model.score(features_valid,target_valid) # calcula la puntuación de accuracy en el conjunto de validación
    if score > best_score:
        best_score = score # guarda la mejor puntuación de accuracy en el conjunto de validación
        best_est = est# guarda el número de estimadores que corresponden a la mejor puntuación de exactitud

print("La exactitud del mejor modelo en el conjunto de validación (n_estimators = {}): {}".format(best_est, best_score))




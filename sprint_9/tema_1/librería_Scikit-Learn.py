#Librería Scikit-Learn

"""
Los algoritmos de aprendizaje suelen ser más complejos que los modelos. 
Entonces, por ahora, piensa en ellos como cajas negras y concéntrate en lo que debes usar como entrada y qué hacer con la salida, en lugar de enfocarte en lo que sucede dentro.
El concepto de caja negra se puede comparar con pedir pizza. Para pedir una pizza, todo lo que tienes que hacer es elegir los ingredientes, dar tu dirección y esperar. 
En realidad, no te importa lo que sucede entre el pedido de la pizza y el repartidor que llama a la puerta. 
Si alguna vez quisieras abrir tu propia pizzería (es decir, hacer un trabajo académico sobre la mejora de algoritmos), necesitarías conocer todo el funcionamiento interno. 
Por ahora, deja la cocina a los profesionales entrenados. ¿Entiendes? ¿Entrenados... como en "modelo entrenado"?
Pero ¿dónde podemos conseguir una de estas cajas negras para entrenar a nuestro modelo? Las librerías de Python ofrecen muchos algoritmos. 
En esta lección trabajaremos con la popular librería scikit-learn o sklearn (kit científico para aprender).
Scikit-learn es una gran fuente de herramientas para trabajar con datos y modelos. 
La librería se divide en módulos para mayor comodidad. Los árboles de decisión se almacenan en el módulo tree.
Cada modelo corresponde a una clase separada en scikit-learn.  
DecisionTreeClassifier es una clase para clasificaciones de árboles de decisión. Vamos a importarla desde la librería:
"""

from sklearn.tree import DecisionTreeClassifier
#Luego creamos una instancia de la clase:
model = DecisionTreeClassifier()

# La variable model ahora almacena nuestro modelo, y tenemos que ejecutar un algoritmo de aprendizaje para entrenar el modelo para hacer predicciones.

#EJERCICIO 1

'''

Comencemos con el entrenamiento del modelo. 
En la Lección 5 guardamos el conjunto de datos de entrenamiento en las variables features y target. Para iniciar el entrenamiento, llama al método fit() y pásale tus variables como argumento.

model.fit(features, target)

Finaliza el código e imprime la variable model en la pantalla (ya en el precódigo).
'''

try:
    import pandas as pd
    from sklearn import set_config
    from sklearn.tree import DecisionTreeClassifier

    # no cambies estos parámetros de configuración
    set_config(print_changed_only=False)
    model = DecisionTreeClassifier()

    # importa el árbol de decisión de la librería sklearn
    # < escribe tu código aquí >

    df = pd.read_csv('/datasets/train_data_us.csv')

    df.loc[df['last_price'] > 113000, 'price_class'] = 1
    df.loc[df['last_price'] <= 113000, 'price_class'] = 0

    features = df.drop(['last_price', 'price_class'], axis=1)
    target = df['price_class']
    model.fit(features, target)
    # crea un modelo vacío y asígnalo a una variable
    # entrena un modelo llamando al método fit()
    # < escribe tu código aquí >

    print(model)
except:
    print("prueba")

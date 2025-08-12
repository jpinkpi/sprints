#Ajuste de hiperparámetros
'''
El hiperparámetro más importante de un árbol de decisión es max_depth. 
Este determina qué obtendremos al final: 
un tocón con una pregunta o un arce con una enorme copa.



¿Cómo podemos encontrar el mejor valor para el hiperparámetro max_depthsi queremos mejorar el modelo? No lo sabemos de antemano. 
Así que iteraremos diferentes valores con un bucle y compararemos la calidad de las diferentes versiones del modelo. 
Lo comprobaremos automáticamente sin el conjunto de prueba.
'''

#EJERCICIO
'''
Cambia el hiperparámetro max_depthen el bucle de 1 a 5. Por cada valor, imprime la calidad para el conjunto de validación. 

Imprime esto en la pantalla:
max_depth = 1 : ...
max_depth = 2 : ...
...
max_depth = 5 : ...

Aún no necesitamos probar nuestros modelos con el dataset de prueba. Primero seleccionaremos el mejor modelo.
'''

try:
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(r"C:\Users\josep\Downloads\moved_train_data_us.csv")
    df.loc[df['last_price'] > 113000, 'price_class'] = 1
    df.loc[df['last_price'] <= 113000, 'price_class'] = 0

    df_train, df_valid = train_test_split(df, test_size=0.25, random_state=12345)

    features_train = df_train.drop(['last_price', 'price_class'], axis=1)
    target_train = df_train['price_class']
    features_valid = df_valid.drop(['last_price', 'price_class'], axis=1)
    target_valid = df_valid['price_class']

    for depth in range(1,6):
        model= DecisionTreeClassifier(random_state=12345, max_depth=depth)
        model.fit(features_train, target_train)
        predictions_valid = model.predict(features_valid)
        print("max_depth =", depth, ": ", end='')
        print(accuracy_score(target_valid, predictions_valid))
except: print("prueba")
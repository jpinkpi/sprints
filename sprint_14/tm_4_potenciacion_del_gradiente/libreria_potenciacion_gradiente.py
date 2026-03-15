#Veamos estas librerías para potencciación del gradiente: XGBoost, LightGBM y CatBoost

'''
Veamos cada una:

XGBoost (potenciación del gradiente extrema) es una librería popular de potenciación del gradiente en Kaggle. Código abierto. Lanzada en 2014.
LightGBM (máquina ligera de potenciación del gradiente). 
Desarrollada por Microsoft. Entrenamiento de potenciación del gradiente rápido y preciso. 
Funciona directamente con características categóricas. 
Lanzada en 2017. Comparación con XGBoost: https://lightgbm.readthedocs.io/en/latest/Experiments.html (materiales en inglés).
CatBoost (potenciación categórica).
Desarrollada por Yandex. Superior a otros algoritmos en términos de métricas de evaluación. 
Aplica varias técnicas de codificación para características categóricas (LabelEncoding, One-Hot Encoding). 
Lanzada en 2017. Comparación con XGBoost y LightGBM: https://catboost.ai/#benchmark (materiales en inglés).

Por ejemplo, así es como CatBoost funciona con características categóricas:
un perro salchicha sigue siendo un perro salchicha y un stafford sigue siendo un stafford. 
No es necesario codificar usando 1 o 0.
'''

'''
Comparemos las librerías en función de dos características:

Librería	Tratamiento de características categóricas	Velocidad
XGBoost	No	Baja
LightGBM	Sí	Alta
CatBoost	Sí	Alta
Echemos un vistazo a la librería CatBoost. También puedes leer sobre LightGBM cuando tengas oportunidad.

Toma el conjunto de datos de la tarea de seguros (del curso de aprendizaje supervisado).
'''
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv('travel_insurance_us.csv')
    print(data.head())

    features_train, features_valid, target_train, target_valid = train_test_split(
        data.drop('Claim', axis=1), data.Claim, test_size=0.25, random_state=12345
    )
except: print("No hay bases de datos ")

'''
Las características categóricas tienen valores perdidos, anotados como valores nuevos. Por ejemplo, None en la columna Género. 
Este no es un problema para CatBoost: dichos valores son categorías separadas para la librería.

Importa CatBoostClassifier de la librería y crea un modelo. 
Ya que tenemos un problema de clasificación, especifica la función de pérdida logística. Haz 10 iteraciones para que no tengamos que esperar demasiado.
'''

from catboost import CatBoostClassifier

model = CatBoostClassifier(loss_function="Logloss", iterations=10)
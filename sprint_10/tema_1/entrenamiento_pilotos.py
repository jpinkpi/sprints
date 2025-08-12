#Entrenamiento de pilotos
'''
Para predecir la clase, necesitamos a nuestra vieja amiga, la regresión logística.
La regresión logística se utiliza para tareas de clasificación. En nuestro caso, es binaria: se presenta el reclamo o no.

Intentaremos entrenar nuestro modelo. ¿Crees que podemos entrenarlo usando los datos sin procesar? ¡Vamos a arriesgarnos!

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('travel_insurance_us.csv')

train, valid = train_test_split(data, test_size=0.25, random_state=12345)

features_train = train.drop('Claim', axis=1)
target_train = train['Claim']
features_valid = valid.drop('Claim', axis=1)
target_valid = valid['Claim']

model = LogisticRegression()
model.fit(features_train, target_train)
...

ValueError: could not convert string to float: 'M'
Muy bien, ¡adelante! Es el momento perfecto para un "te lo dije". Tenías razón, cometimos un error.

¿Qué salió mal?

La regresión logística determina la categoría utilizando una fórmula que consta de características numéricas. Además de numéricas, nuestros datos contenían características categóricas, de ahí el error.
'''

#EJERCICIO 
'''
Verifica los tipos de características almacenadas en la tabla. 
Muéstralos. Luego muestra en pantalla los primeros cinco valores de la columna Gender
'''
try:
    import pandas as pd

    data = pd.read_csv('/datasets/travel_insurance_us.csv')
    print(data.dtypes)
    print(data["Gender"].head())
    # < escribe tu código aquí >
except:print("prueba")

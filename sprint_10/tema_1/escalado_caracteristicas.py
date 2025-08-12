#Escalado de características
'''
¿Qué debemos hacer si las características tienen diferentes escalas? ¡Deberíamos estandarizar!
Echemos un vistazo más de cerca a las columnas Age y Commission (in value). Imagina que para Age los valores posibles están en el rango de 0 a 100. Para Commission (in value), los valores son de $100 a $1000. La magnitud de los valores y la dispersión es mayor para la columna Commission (in value), lo que significa que el algoritmo encontrará que la característica Commission (in value) es más importante que la de Age. No queremos eso. Todas las características deben considerarse igualmente importantes antes de la ejecución del algoritmo.

Este problema se puede solucionar con el escalado de características. 

Una de las formas de escalar las características es estandarizar los datos.

Piensa que todas las características se distribuyen normalmente, la media (M) y la varianza (Var) se determinan a partir de la muestra. 
'''

'''
Para la nueva característica, la media se convierte en 0 y la varianza es igual a 1.
Hay una clase sklearn dedicada para la estandarización de datos que se llama StandardScaler. Está en el módulo sklearn.preprocessing.
Importa StandardScaler de la librería:
'''
from sklearn.preprocessing import StandardScaler

#Crea una instancia de la clase y ajústala usando los datos de entrenamiento. El proceso de ajuste implica calcular la media y la varianza:
features_train = "prueba"
features_valid = "prueba"
scaler = StandardScaler()
scaler.fit(features_train)

#Transforma el conjunto de entrenamiento y el conjunto de validación usando transform(). 
#Almacena los conjuntos modificados en variables de la siguiente manera: features_train_scaled y features_valid_scaled:

features_train_scaled = scaler.transform(features_train)
features_valid_scaled = scaler.transform(features_valid)

#EJERCICIO
'''
Estandariza las características numéricas. Importa StandardScaler desde el módulo sklearn.preprocessing.
Crea una instancia de la clase StandardScaler() y ajústala usando los datos de entrenamiento. 
(La variable numeric ya contiene la lista de todas las características numéricas).
Almacena los conjuntos de entrenamiento y validación modificados en variables de la siguiente manera: features_train y features_valid.
Muestra en pantalla las primeras cinco filas.
Cuando estandarices las funciones, es posible que te encuentres con SettingWithCopyWarning. Esta no te impedirá pasar la tarea, pero si lo deseas, puedes silenciarla con la siguiente declaración:
pd.options.mode.chained_assignment = None
'''
try:
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    pd.options.mode.chained_assignment = None
    # < escribe tu código aquí >

    data = pd.read_csv('/datasets/travel_insurance_us.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    numeric = ['Duration', 'Net Sales', 'Commission (in value)', 'Age']

    scaler = StandardScaler()
    scaler.fit(features_train[numeric])
    features_train[numeric] = scaler.transform(features_train[numeric])
    features_valid[numeric] = scaler.transform(features_valid[numeric])
except:print("Prueba")
# < escribe tu código aquí >



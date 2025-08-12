#Carga de datos
'''
Analicemos el ejercicio, carguemos los datos e investiguemos las características.
Los ingresos de los pagos de la clientela superan con creces los costos de liquidación de reclamaciones. El plan de cada compañía de seguros es que solo una pequeña parte de la clientela hará un reclamo, mientras que la mayoría de las personas no recurrirán a él.

Cada cliente que se une es una caja negra para la compañía de seguros de viaje. ¿Qué probabilidad hay de que la empresa tenga que pagar? Históricamente, solo el 1 % de la clientela reclama beneficios de seguro. ¿Cómo predecir, entonces, si una persona lo necesitará?

El machine learning nos permite calcular la probabilidad de una reclamación de seguro.

Tenemos las siguientes características:

Agency: nombre de la agencia de seguros
Agency Type: tipo de agencia de seguros
Distribution Channel: canal de distribución de la agencia de seguros
Product Name: nombre del producto de seguro
Duration: duración del viaje (días)
Destination: destino del viaje
Net Sales: ventas netas ($)
Commission: comisión de la agencia de seguros ($)
Gender: género de la persona asegurada
Age: edad de la persona asegurada
Objetivo:

Claim — reclamación de liquidación (1 es sí, 0 es no)
'''

#EJERCICIO 1
'''Carga los datos de /datasets/travel_insurance_us.csv a la variable data. 
Muestra en la pantalla los primeros diez elementos. Mira los datos.'''
try:
    import pandas as pd
    data= pd.read_csv("/datasets/travel_insurance_us.csv")
    print(data.head(10))# < escribe tu código aquí  >
except:print("prueba")




#EJERCICIO 2
'''
.

Divide los datos en dos conjuntos:

conjunto de entrenamiento (train)
conjunto de validación (valid) — 25% de los datos de origen
Especifica random_state=12345. Declara cuatro variables y almacena las características y el objetivo de la siguiente manera:

características: features_train, features_valid
objetivo: target_train, target_valid
Muestra en pantalla los tamaños de las tablas almacenadas en las variables: features_train y features_valid.
'''


try:

    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv('/datasets/travel_insurance_us.csv')
    df_train, df_valid = train_test_split(data, test_size =0.25, random_state=12345 )

    features_train = df_train.drop(["Claim"], axis=1)
    target_train = df_train["Claim"]
    features_valid = df_valid.drop(["Claim"],axis=1)
    target_valid = df_valid["Claim"]
    print(features_train.shape)
    print(features_valid.shape)
    # < escribe tu código aquí >
except:print("prueba")

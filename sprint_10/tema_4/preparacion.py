#Preparación de datos

'''
Veamos la tarea, carguemos los datos e investiguemos sus características.
Cada año crece la carga de infraestructura de las aerolíneas. Por ejemplo, en diciembre de 2018, el Aeropuerto Internacional de Bombay estableció un récord mundial con 1007 aviones que completaron el despegue y el aterrizaje ¡en 24 horas!

Los vuelos retrasados ponen en peligro el funcionamiento del aeropuerto, perjudicando los ingresos tanto del aeropuerto como de la aerolínea.

Para resolver este problema, entrenaremos un modelo de regresión para predecir los tiempos de retraso de los vuelos en minutos.



Descripción de datos
Tenemos las siguientes características:

Month: mes de vuelo
Day: fecha del vuelo
Day Of Week: día de la semana del vuelo
Airline: nombre de la aerolínea
Origin Airport Delay Rate: tasa de retraso de vuelo en el aeropuerto de origen
Destination Airport Delay Rate: tasa de retraso del vuelo en el aeropuerto de destino
Scheduled Time: tiempo de vuelo programado
Distance: distancia del vuelo
Scheduled Departure Hour: hora de salida programada (hora)
Scheduled Departure Minute: hora de salida programada (minuto)
Objetivo:

Arrival Delay: retraso de llegada (minuto)

'''

#Ejercicio 1 
'''
Carga los datos de flights.csv. Se debe mostrar lo siguiente:

Tamaño de la tabla
Primeras cinco filas de la tabla
Mira los datos.

'''

try:
    import pandas as pd
    data = pd.read_csv('/datasets/flights.csv')
    print(data.shape)
    print(data.head())
    # < escribe el código aquí  >

except:print("prueba")

#Ejercicio 2

'''
Codifica la característica categórica utilizando OHE. Estandariza las características numéricas. 
Muestra los tamaños de la tabla (en precódigo).
Cuando muestres los resultados en pantalla, no prestes atención al texto en color rojo. Está bien.

'''

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = pd.read_csv('/datasets/flights.csv')

    # <  aplica one hot encoding y evita la trampa de las variables ficticias encodifica los datos y evita la trampa de las variables ficticias >
    data_ohe = pd.get_dummies(data, drop_first=True)
    target = data_ohe['Arrival Delay']
    features = data_ohe.drop(['Arrival Delay'], axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    numeric = ['Day', 'Day Of Week', 'Origin Airport Delay Rate',
        'Destination Airport Delay Rate', 'Scheduled Time', 'Distance',
        'Scheduled Departure Hour', 'Scheduled Departure Minute']

    scaler = StandardScaler()
    scaler.fit(features_train[numeric])
    features_train[numeric] = scaler.transform(features_train[numeric])
    features_valid[numeric] = scaler.transform(features_valid[numeric])
    # <  transforma el conjunto de validación >

    print(features_train.shape)
    print(features_valid.shape)

except:print("preuba")



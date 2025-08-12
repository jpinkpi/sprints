#La trampa dummy
'''
Como es de esperar, en One-Hot Encoding hay más de lo que parece. 
Cuando los datos son abundantes, tenemos la posibilidad de caer en la trampa de las características dummy. 
Aquí aprenderemos cómo evitarla.

Para obtener tu identificación del Departamento de Vehículos Motorizados, debes presentar un comprobante de residencia actual en tu estado. 
Tú sabes cómo es el departamento, así que para no fallar y terminar en una sola visita, traes la factura de servicios públicos, 
el contrato de hipoteca y la factura del impuesto a la propiedad, aunque sería suficiente traer dos de esos documentos. 
Si bien traer todos los documentos que puedas para luchar contra la burocracia puede ser una buena idea al tratar con el deparmento, no se aplica el mismo principio a la capacitación de modelos. 
Si mantienes las características como están ahora, esto dificultará el proceso de entrenamiento.

Hemos agregado tres columnas nuevas a nuestra tabla, pero su alta correlación confundirá a nuestro modelo. 
Para evitar esto, podemos eliminar con seguridad cualquier columna, ya que sus valores se pueden deducir fácilmente de las otras dos columnas 
(tiene 1 donde las otras dos columnas tienen ceros y tiene ceros en el resto). 
De esta manera, no caeremos en la trampa dummy.


Para eliminar la columna, llama a la función pd.get_dummies() junto con el parámetro drop_first. 
Si pasas drop_first=True entonces se elimina la primera columna. De lo contrario, es drop_first=False por defecto y no se descartan columnas.
'''

#EJERCICIO 1
'''
Programa la variableGender con OHE. 
Llama a pd.get_dummies() con el argumento drop_first para evitar la trampa dummy.
Muestra en pantalla las primeras cinco filas de la tabla modificada.
'''


try:
    import pandas as pd

    data = pd.read_csv('/datasets/travel_insurance_us.csv')
    print(pd.get_dummies(data ["Gender"], drop_first= True).head(5))
    # < escribe el código aquí  >
except:print("prueba")



#EJERCICIO 2
'''
Programa todo el DataFrame con One-Hot. 
Llama a pd.get_dummies() con el argumento drop_first. 
Almacena la tabla en la variable data_ohe.
Muestra en pantalla las primeras tres filas de la tabla resultante.
'''
try:
    import pandas as pd

    data = pd.read_csv('/datasets/travel_insurance_us.csv')
    data_ohe = pd.get_dummies(data, drop_first=True)
    print(data_ohe.head(3))
    # < escribe el código aquí  >
except:print("prueba")


#EJERCICIO 3
'''
Divide los datos de origen en dos conjuntos utilizando la proporción de 75:25 (%):

entrenamiento (train)
validación (valid)
Declara cuatro variables y almacena las características y el objetivo de la siguiente manera:

características: features_train, features_valid
objetivo: target_train, target_valid
Vas a dominar una forma alternativa de usar la función train_test_split(): puede tomar dos variables (características y objetivo). 
Consulta la documentación de este método en particular.

Entrena una regresión logística. Muestra en pantalla el texto "¡Entrenado!" (ya en precódigo) para asegurarte de que el código terminó de ejecutarse sin problemas.

Especifica random_state=12345 tanto para la división de datos como para el entrenamiento del modelo.

Al entrenar una regresión logística, es posible que encuentres una advertencia de sklearn. Esta no te impedirá pasar la tarea, pero si no te gusta ver un montón de código en rojo en tu pantalla, especifica el argumento solver='liblinear'; librería linear:

model = LogisticRegression(solver='liblinear')
'''

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    data = pd.read_csv('/datasets/travel_insurance_us.csv')

    data_ohe = pd.get_dummies(data, drop_first=True)
    target = data_ohe['Claim']
    features = data_ohe.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=.25, random_state=12345) 
    model = LogisticRegression(solver="liblinear", random_state=12345)
    model.fit(features_train, target_train)
    # < escribe el código aquí  >

    print('¡Entrenado!')

except:print("prueba")

#Ejercicio 3

'''
Crea un modelo constante: predice la clase "0" para cualquier observación. 
Guarda sus predicciones en la variable target_pred_constant. Muestra en pantalla el valor de accuracy.
'''

try:
    import pandas as pd
    from sklearn.metrics import accuracy_score

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    target_pred_constant= pd.Series(0, index=target.index)
    print(accuracy_score(target, target_pred_constant))
    # <escribe el código aquí >

except:print("prueba")




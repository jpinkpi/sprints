#Codificación One-Hot
'''
Existe una técnica especial para transformar características categóricas en numéricas. Se llama One-Hot Encoding (OHE).
¿Cómo funciona One-Hot Encoding? Vamos a utilizar los valores característicos de Gender

 Vamos a crear una columna separada para cada valor de Gender (F, M, Ninguno):

Gender_F
Gender_M
Gender_None
La columna que obtiene un 1 depende del valor de la característica "Gender". Si el valor es F, 1 va a la columna Gender_F; si es M, entonces 1 va a Gender_M.

Recapitulemos. La técnica OHE nos permite transformar características categóricas en características numéricas en dos pasos:

Agrega una columna separada para cada valor de función.
Si la categoría se ajusta a la observación, se asigna 1, de lo contrario, se asigna 0.
Las nuevas columnas (Gender_F, Gender_M, Gender_None) se denominan variables dummy.

La librería pandas tiene una función pd.get_dummies() que se puede usar para obtener variables dummy.
'''

#EJERCICIO 
'''
Podemos mirar los valores de la columna Gender usando OHE. 
Llama a pd.get_dummies() y muestra en pantalla las primeras cinco filas de la tabla transformada.
'''
try:
    import pandas as pd

    data = pd.read_csv('/datasets/travel_insurance_us.csv')
    print(pd.get_dummies(data["Gender"]).head())
    # < escribe tu código aquí >
except:print("prueba")


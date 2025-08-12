#Recall

'''
La matriz de confusión te ayudará a crear nuevas métricas. Comencemos con recall.
Recall revela la porción de respuestas positivas identificadas por el modelo o la proporción de respuestas positivas marcadas como positivas por el modelo (VP) 
frente a las respuestas positivas marcadas como positivas por el modelo (VP) más las respuestas marcadas como negativas por el modelo que en realidad son positivas (FN). 
Estas respuestas positivas son valiosas, por lo que es importante saber con qué eficacia las encuentra el modelo.

Recall se calcula usando esta fórmula:

                                        recall = VP/(VP + FN)
'''
'''
Veamos un ejemplo:

100 personas aseguradas hicieron reclamaciones. Este es el número de todas las observaciones positivas o VP + FN,
El modelo identificó correctamente solo 20
entonces, recall es 0.2.
Recall es la proporción de VP entre todas las respuestas que tienen una etiqueta verdadera de 1. Queremos que el valor de recall esté cerca de 1. 
Esto significaría que el modelo es bueno para identificar verdaderos positivos. Si está más cerca de cero, el modelo necesita ser revisado y arreglado.
'''


#EJERCICIO 
'''
En el módulo sklearn.metrics, encuentra la función que se encarga de calcular recall. Impórtala.

La función toma respuestas y predicciones correctas y devuelve la proporción de respuestas correctas encontradas por el modelo. 
Muestra los resultados en la pantalla.
'''

try:
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import recall_score# <write code here>
    from sklearn.model_selection import train_test_split

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = DecisionTreeClassifier(random_state=12345)
    model.fit(features_train, target_train)
    predicted_valid = model.predict(features_valid)

    rcc = recall_score(target_valid, predicted_valid)
    print(rcc)
except:print("prueba")
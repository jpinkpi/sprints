#Precisión
'''
Otra métrica para evaluar la calidad de una predicción de clase objetivo es la precisión.
La precisión mide la relación entre las predicciones verdaderas positivas y todas las predicciones positivas realizadas por el modelo. 
Por lo tanto, cuantas más predicciones de falsos positivos se hagan, menor será la precisión.

Veamos un ejemplo:

Según la predicción del modelo, 100 personas aseguradas solicitarán una indemnización. Este es el número de todas las observaciones que el modelo ha etiquetado como positivas, o VP + FP
20 de ellas realmente solicitaron un pago de seguro (este es el número de VP);
la precisión es 0.2.
Recuerda que VP representa respuestas verdaderas positivas. FP representa respuestas positivas marcadas por el modelo. 
Necesitamos que la precisión esté cerca de uno.
'''


#EJERCICIO 1
'''
En el módulo sklearn.metrics, encuentra la función que calcula la precisión. Impórtala.
Esta función toma respuestas y predicciones correctas. 
Devuelve observaciones marcadas como positivas por el modelo que en realidad son positivas. Muestra los resultados en la pantalla
'''

try:
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import precision_score # < write code here >
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
    precision_score = precision_score(target_valid, predicted_valid)
    print(precision_score)

except:print("prueba")
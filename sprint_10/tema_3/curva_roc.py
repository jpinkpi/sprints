#Curva ROC
'''
Hemos observado un enfrentamiento entre la Tasa de Verdaderos Positivos (TVP) y la Tasa de Falsos Positivos (TFP). 
Ahora, vamos a visualizar esta relación trazando la curva ROC.
Colocamos los valores de la tasa de falsos positivos (TFP) a lo largo del eje horizontal 
y los valores de la tasa de verdaderos positivos (TVP) a lo largo del eje vertical. 
Luego iteramos los valores del umbral de regresión logística y trazamos una curva. 
Se llama la curva ROC (del inglés, Característica Operativa del Receptor,* un término de la teoría del procesamiento de señales).

Para un modelo que siempre responde aleatoriamente, 
la curva ROC es una línea diagonal que va desde la esquina inferior izquierda hasta la esquina superior derecha. 
Cuanto más se aleje la curva ROC de esta línea diagonal hacia la esquina superior izquierda, mejor será el modelo, ya que indica una mayor relación TVP-TFP. 

ara encontrar cuánto difiere nuestro modelo del modelo aleatorio, 
calculemos el valor AUC-ROC (Área Bajo la Curva ROC). 
Esta es una nueva métrica de evaluación con valores en el rango de 0 a 1. 
El valor AUC-ROC para un modelo aleatorio es 0.5.

Podemos trazar una curva ROC con la variable roc_curve() del módulo sklearn.metrics:

                     from sklearn.metrics import roc_curve

Esta toma los valores objetivo y las probabilidades de clase positivas, 
supera diferentes umbrales y devuelve tres listas: valores TFP, valores TVP y los umbrales que superó.    

                    fpr, tpr, thresholds = roc_curve(target, probabilities)
'''
#EJERCICIO 1
'''
Haz una curva ROC para la regresión logística y trázala en la gráfica. 
Sigue las instrucciones en el precódigo.
También agregamos la curva ROC del modelo aleatorio al precódigo.
'''


try:
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = LogisticRegression(random_state=12345, solver='liblinear')
    model.fit(features_train, target_train)

    probabilities_valid = model.predict_proba(features_valid)
    probabilities_one_valid = probabilities_valid[:, 1]

    fpr, tpr, thresholds = roc_curve(target_valid, probabilities_one_valid)# < escribe el código aquí >

    plt.figure()

    # < traza la gráfica >

    # Curva ROC para modelo aleatorio (parece una línea recta)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    # < utiliza las funciones plt.xlim() y plt.ylim() para
    #   establecer el límite para los ejes de 0 a 1 >
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel("Tasa de verdaderos positivos")
    plt.xlabel("Tasa de falsos positivos")
    # < utiliza las funciones plt.xlabel() y plt.ylabel() para
    #   nombrar los ejes "Tasa de falsos positivos" y "Tasa de verdaderos positivos">

    # < agrega el encabezado "Curva ROC" con la función plt.title() >
    plt.title("Curva ROC")
    plt.show()
except:print("prueba")

#EJERCICIO 2
'''
Calcula el AUC-ROC para la regresión logística. Encuentra la función adecuada en la documentación de sklearn, 
así como la descripción de cómo funciona. Importa la función. Muestra en la pantalla el valor AUC-ROC.
'''

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    # < write code here >

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = LogisticRegression(random_state=12345, solver='liblinear')
    model.fit(features_train, target_train)

    probabilities_valid = model.predict_proba(features_valid)
    probabilities_one_valid = probabilities_valid[:, 1]
    auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
    # < escribe el código aquí >

    print(auc_roc)
except:print("prueba")



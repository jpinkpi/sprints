#Curva PR

'''
Tracemos los valores de las métricas y veamos cómo responde la curva al cambio de umbral.
En la gráfica, el valor de precisión se traza verticalmente y recall, horizontalmente. 
Una curva trazada a partir de los valores de Precisión y Recall se denomina curva PR. 
Cuanto más alta sea la curva, mejor será el modelo.
'''
# ¿Quieres saber cómo se obtiene la curva? Aquí tienes el código:
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_curve
    from sklearn.linear_model import LogisticRegression

    data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

    target = data['Claim']
    features = data.drop('Claim', axis=1)
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345
    )

    model = LogisticRegression(random_state=12345, solver='liblinear')
    model.fit(features_train, target_train)

    probabilities_valid = model.predict_proba(features_valid)
    precision, recall, thresholds = precision_recall_curve(
        target_valid, probabilities_valid[:, 1]
    )

    plt.figure(figsize=(6, 6))
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.show()
except:print("prueba")
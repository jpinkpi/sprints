#Proyecto Sprint_10
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import numpy as np


df = pd.read_csv(r"C:\Users\josep\Downloads\Churn (1).csv")
print(df.sample(n=6))
df.info()


df["Tenure"] = df["Tenure"].fillna(0).astype("int")

#Vemos las dispersion de la columna de 

class_frequency= df["Exited"].value_counts(normalize=True)
print(class_frequency)


#Hay un error en la columna surname con el nombre Ch'ien

target = df["Exited"]

features = df.drop(["Exited"], axis=1)
features = pd.get_dummies(features)
features.info()


#Dividimos el conjunto de entrenamiento, validacion y prueba

features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, train_size=0.6, random_state=54321
)

features_valid, features_test, target_valid, target_test = train_test_split(
    features_valid, target_valid, test_size=0.5, random_state=54321
)






#Examina el equilibrio de clases. 
#Entrena el modelo sin tener en cuenta el desequilibrio. Describe brevemente tus hallazgos.


#Regresion Logistica (sin contar el desbalance de calses)
model = LogisticRegression(random_state=12345, solver="liblinear")
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
f1 = f1_score(target_valid,predicted_valid)
print("Valor de f1 utilizando Regresión Logística sin darle balance a las clases",f1)

'''Me da un valor muy bajo, por lo que opte por balancear el peso de clase'''
print()



#Mejora la calidad del modelo. Asegúrate de utilizar al menos dos enfoques para corregir el desequilibrio de clases. 
#Utiliza conjuntos de entrenamiento y validación para encontrar el mejor modelo y el mejor conjunto de parámetros. 
#Entrena diferentes modelos en los conjuntos de entrenamiento y validación.
#Encuentra el mejor. Describe brevemente tus hallazgos.


print("regresion logistica")
#Regresion Logistica
model = LogisticRegression(random_state=12345, solver="liblinear", class_weight='balanced')
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
f1 = f1_score(target_valid,predicted_valid)

print("Valor de f1 utilizando Regresión Logística",f1)
#ajuste de humbral
best_f1 = 0
best_threshold = 0
probs_valid = model.predict_proba(features_valid)[:, 1]

for threshold in np.arange(0.1, 0.9, 0.01):
    preds = probs_valid > threshold
    f1 = f1_score(target_valid, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Mejor F1 : {best_f1:.4f} usando threshold = {best_threshold:.2f}")

#Sobremuestreo 

def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345
    )

    return features_upsampled, target_upsampled


features_upsampled, target_upsampled = upsample(
    features_train, target_train, 70
)
model = LogisticRegression(random_state=12345,solver='liblinear', class_weight="balanced")
model.fit(features_upsampled, target_upsampled)
predicted_valid = model.predict(features_valid)

print('F1(sobremuestreo):', f1_score(target_valid, predicted_valid))

#ajuste de humbral
best_f1 = 0
best_threshold = 0
probs_valid = model.predict_proba(features_valid)[:, 1]

for threshold in np.arange(0.1, 0.9, 0.01):
    preds = probs_valid > threshold
    f1 = f1_score(target_valid, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Mejor F1 con sobremuestreo: {best_f1:.4f} usando threshold = {best_threshold:.2f}") 


#Submuestreo

def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)]
        + [features_ones]
    )
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)]
        + [target_ones]
    )

    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345
    )

    return features_downsampled, target_downsampled


features_downsampled, target_downsampled = downsample(
    features_train, target_train, 0.55
)
model = LogisticRegression(random_state=12345, solver="liblinear",class_weight='balanced')
model.fit(features_downsampled, target_downsampled)
predicted_valid = model.predict(features_valid)

print('F1(submuestreo):', f1_score(target_valid, predicted_valid))

#ajuste de humbral
best_f1 = 0
best_threshold = 0
probs_valid = model.predict_proba(features_valid)[:, 1]

for threshold in np.arange(0.1, 0.9, 0.01):
    preds = probs_valid > threshold
    f1 = f1_score(target_valid, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Mejor F1 con subemuestreo: {best_f1:.4f} usando threshold = {best_threshold:.2f}") 


print()
print()
#bosque aleatorio
print("bosque aleatorio")
best_score = 0
best_est = 0
for est in range(1, 51,10): 
    model = RandomForestClassifier(random_state=54321, n_estimators= est,class_weight='balanced') 
    model.fit(features_train,target_train)
    predictions_valid = model.predict(features_valid)
    score =f1_score(target_valid, predictions_valid)
    if score > best_score:
        best_score = score 
        best_est = est 

print("el f1 del mejor modelo utlizando un bosque aleatorio en el conjunto de validación (n_estimators = {}): {}".format(best_est, best_score))
#ajuste de humbral
best_f1 = 0
best_threshold = 0
probs_valid = model.predict_proba(features_valid)[:, 1]

for threshold in np.arange(0.1, 0.9, 0.01):
    preds = probs_valid > threshold
    f1 = f1_score(target_valid, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Mejor F1: {best_f1:.4f} usando threshold = {best_threshold:.2f}") 


#Sobremuestreo

model = RandomForestClassifier(random_state=12345, n_estimators=55,class_weight='balanced')
model.fit(features_upsampled, target_upsampled)
predicted_valid = model.predict(features_valid)
print('F1(sobremuestreo):', f1_score(target_valid, predicted_valid))
#ajuste de humbral
best_f1 = 0
best_threshold = 0
probs_valid = model.predict_proba(features_valid)[:, 1]

for threshold in np.arange(0.1, 0.9, 0.01):
    preds = probs_valid > threshold
    f1 = f1_score(target_valid, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Mejor F1 con sobremuestreo: {best_f1:.4f} usando threshold = {best_threshold:.2f}") 


#Submuestreo
model = RandomForestClassifier(random_state=12345, n_estimators=55,class_weight='balanced')
model.fit(features_downsampled, target_downsampled)
predicted_valid = model.predict(features_valid)
print('F1(submuestreo):', f1_score(target_valid, predicted_valid))
#ajuste de humbral
best_f1 = 0
best_threshold = 0
probs_valid = model.predict_proba(features_valid)[:, 1]

for threshold in np.arange(0.1, 0.9, 0.01):
    preds = probs_valid > threshold
    f1 = f1_score(target_valid, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Mejor F1 con subemuestreo: {best_f1:.4f} usando threshold = {best_threshold:.2f}") 
print()











print("Arbol de decisión")
#Árbol de decision
best_score = 0
best_depth = 0

for depth in range(1,20):
        model= DecisionTreeClassifier(random_state=12345, max_depth=depth,class_weight='balanced')
        model.fit(features_train, target_train)
        predictions_valid = model.predict(features_valid)
        score = f1_score(target_valid, predictions_valid)

        if score > best_score:
                    best_score = score 
                    best_depth = depth
print("el f1 del mejor modelo utlizando un arbol de decisión en el conjunto de validación (max_depth = {}): {}".format(best_depth, best_score))

#ajuste de humbral
best_f1 = 0
best_threshold = 0

for threshold in np.arange(0.1, 0.9, 0.01):
    preds = probs_valid > threshold
    f1 = f1_score(target_valid, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Mejor F1 : {best_f1:.4f} usando threshold = {best_threshold:.2f}") 


#Sobremuestro

model = DecisionTreeClassifier(random_state=12345, max_depth=9 ,class_weight='balanced')
model.fit(features_upsampled, target_upsampled)
predicted_valid = model.predict(features_valid)
print('F1(sobremuestreo):', f1_score(target_valid, predicted_valid))
probs_valid = model.predict_proba(features_valid)[:, 1]
#ajuste de humbral
best_f1 = 0
best_threshold = 0

for threshold in np.arange(0.1, 0.9, 0.01):
    preds = probs_valid > threshold
    f1 = f1_score(target_valid, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Mejor F1 con sobremuestreo: {best_f1:.4f} usando threshold = {best_threshold:.2f}")       

#Submuestreo
model = DecisionTreeClassifier(random_state=12345, max_depth=9 ,class_weight='balanced')
model.fit(features_downsampled, target_downsampled)
predicted_valid = model.predict(features_valid)
print('F1(submuestreo):', f1_score(target_valid, predicted_valid))
#ajuste de humbral
best_f1 = 0
best_threshold = 0
probs_valid = model.predict_proba(features_valid)[:, 1]
for threshold in np.arange(0.1, 0.9, 0.01):
    preds = probs_valid > threshold
    f1 = f1_score(target_valid, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Mejor F1 con subemuestreo: {best_f1:.4f} usando threshold = {best_threshold:.2f}")      



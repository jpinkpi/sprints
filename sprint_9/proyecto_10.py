#Proyecto Sprint 10
import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from matplotlib import pyplot as plt


#Funciones Utilizadas
def buscar_mejor_umbral(probs_valid, y_valid):
    best_f1 = 0
    best_threshold = 0
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds = probs_valid > threshold
        f1 = f1_score(y_valid, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_f1, best_threshold

def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    return shuffle(features_upsampled, target_upsampled, random_state=12345)

def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = target[target == 1]
    features_downsampled = pd.concat([
        features_zeros.sample(frac=fraction, random_state=12345),
        features[target == 1]
    ])
    target_downsampled = pd.concat([
        target[target == 0].sample(frac=fraction, random_state=12345),
        target[target == 1]
    ])
    return shuffle(features_downsampled, target_downsampled, random_state=12345)

#Carga y procesamiento


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

print("-" * 60)
print("Regresión logística sin balancear clases")

model = LogisticRegression(random_state=12345, solver="liblinear")
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
f1 = f1_score(target_valid,predicted_valid)
print("Valor de f1 utilizando Regresión Logística sin darle balance a las clases",f1)

'''Me da un valor muy bajo, por lo que opte por balancear el peso de clase'''
print()


print("\nRegresión logística con balanceo de clases")
model = LogisticRegression(random_state=12345, solver="liblinear", class_weight="balanced")
model.fit(features_train, target_train)
probs_valid = model.predict_proba(features_valid)[:, 1]
best_f1, best_thresh = buscar_mejor_umbral(probs_valid, target_valid)
print(f"Mejor F1: {best_f1:.4f} con threshold = {best_thresh:.2f}")

# Sobremuestreo
print("\nSobremuestreo con Regresión Logística")
features_up, target_up = upsample(features_train, target_train, 70)
model.fit(features_up, target_up)
probs_valid = model.predict_proba(features_valid)[:, 1]
best_f1, best_thresh = buscar_mejor_umbral(probs_valid, target_valid)
print(f"Mejor F1 (upsample): {best_f1:.4f} con threshold = {best_thresh:.2f}")

# Submuestreo
print("\nSubmuestreo con Regresión Logística")
features_down, target_down = downsample(features_train, target_train, 0.55)
model.fit(features_down, target_down)
probs_valid = model.predict_proba(features_valid)[:, 1]
best_f1, best_thresh = buscar_mejor_umbral(probs_valid, target_valid)
print(f"Mejor F1 (downsample): {best_f1:.4f} con threshold = {best_thresh:.2f}")

# ------------------------------------
# BOSQUE ALEATORIO
# ------------------------------------
print("\n" + "-" * 60)
print("Bosque Aleatorio")
best_score = 0
best_est = 0

for est in range(1, 51, 10):
    model = RandomForestClassifier(random_state=54321, n_estimators=est, class_weight='balanced')
    model.fit(features_train, target_train)
    preds = model.predict(features_valid)
    score = f1_score(target_valid, preds)
    if score > best_score:
        best_score = score
        best_est = est

print(f"Mejor F1 (train): {best_score:.4f} con n_estimators = {best_est}")
model = RandomForestClassifier(random_state=54321, n_estimators=best_est, class_weight='balanced')
model.fit(features_train, target_train)
probs_valid = model.predict_proba(features_valid)[:, 1]
best_f1, best_thresh = buscar_mejor_umbral(probs_valid, target_valid)
print(f"Mejor F1 (threshold): {best_f1:.4f} con threshold = {best_thresh:.2f}")

# Sobremuestreo
print("\nSobremuestreo con Bosque Aleatorio")
features_up, target_up = upsample(features_train, target_train, 70)
model.fit(features_up, target_up)
probs_valid = model.predict_proba(features_valid)[:, 1]
best_f1, best_thresh = buscar_mejor_umbral(probs_valid, target_valid)
print(f"Mejor F1 (upsample): {best_f1:.4f} con threshold = {best_thresh:.2f}")

# Submuestreo
print("\nSubmuestreo con Bosque Aleatorio")
features_down, target_down = downsample(features_train, target_train, 0.55)
model.fit(features_down, target_down)
model.fit(features_down, target_down)
probs_valid = model.predict_proba(features_valid)[:, 1]
best_f1, best_thresh = buscar_mejor_umbral(probs_valid, target_valid)
print(f"Mejor F1 (downsample): {best_f1:.4f} con threshold = {best_thresh:.2f}")

# ------------------------------------
# ÁRBOL DE DECISIÓN
# ------------------------------------
print("\n" + "-" * 60)
print("Árbol de Decisión")
best_score = 0
best_depth = 0

for depth in range(1, 20):
    model = DecisionTreeClassifier(random_state=12345, max_depth=depth, class_weight="balanced")
    model.fit(features_train, target_train)
    preds = model.predict(features_valid)
    score = f1_score(target_valid, preds)
    if score > best_score:
        best_score = score
        best_depth = depth

print(f"Mejor F1 (train): {best_score:.4f} con max_depth = {best_depth}")
model = DecisionTreeClassifier(random_state=12345, max_depth=best_depth, class_weight="balanced")
model.fit(features_train, target_train)
probs_valid = model.predict_proba(features_valid)[:, 1]
best_f1, best_thresh = buscar_mejor_umbral(probs_valid, target_valid)
print(f"Mejor F1 (threshold): {best_f1:.4f} con threshold = {best_thresh:.2f}")

# Sobremuestreo
print("\nSobremuestreo con Árbol de Decisión")
features_up, target_up = upsample(features_train, target_train, 70)
model.fit(features_up, target_up)
probs_valid = model.predict_proba(features_valid)[:, 1]
best_f1, best_thresh = buscar_mejor_umbral(probs_valid, target_valid)
print(f"Mejor F1 (upsample): {best_f1:.4f} con threshold = {best_thresh:.2f}")

# Submuestreo
print("\nSubmuestreo con Árbol de Decisión")
features_down, target_down = downsample(features_train, target_train, 0.55)
model.fit(features_down, target_down)
probs_valid = model.predict_proba(features_valid)[:, 1]
best_f1, best_thresh = buscar_mejor_umbral(probs_valid, target_valid)
print(f"Mejor F1 (downsample): {best_f1:.4f} con threshold = {best_thresh:.2f}") #GANADORA 



#Prueba Final
probs_test = model.predict_proba(features_test)[:,1]
preds_test = probs_test> best_thresh

f1_test = f1_score(target_test, preds_test)
precision_test = precision_score(target_test, preds_test)
recall_test = recall_score(target_test, preds_test)
accuracy_test = accuracy_score(target_test, preds_test)
auc_roc = roc_auc_score(target_test, preds_test)

print("\n" + "-" * 60)
print("Evaluacion en el conjunto de PREUBA:")
print(f"F1 score        :{f1_test:.4f}")
print(f"Precisión       :{precision_test:4f}")
print(f"Recall          :{recall_test:4f}")
print(f"Exactitud       :{accuracy_test:4f}")
print(f"AUC_ROC          :{auc_roc:4f}")

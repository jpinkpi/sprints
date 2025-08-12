#Sprint 12 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor


df_train = pd.read_csv(r"C:\Users\josep\Downloads\gold_recovery_train.csv")
df_test = pd.read_csv(r"C:\Users\josep\Downloads\gold_recovery_test.csv")
df_full = pd.read_csv(r"C:\Users\josep\Downloads\gold_recovery_full.csv")



'''
1.2. Comprueba que el cálculo de la recuperación sea correcto. 
Calcula la recuperación de la característica rougher.output.recovery 
mediante el conjunto de entrenamiento. 
Encuentra el EAM entre tus cálculos y los valores de la característica. 
Facilita los resultados.
'''
C = df_train['rougher.output.concentrate_au']
F = df_train["rougher.input.feed_au"]
T = df_train['rougher.output.tail_au']

mask = (F.notna()) & (C.notna()) & (T.notna()) &  (C !=F)
recuperacion = np.where(mask, (C*(F-T)) / (F*(C-T)) * 100, np.nan)


real = df_train.loc[mask, "rougher.output.recovery"]
mae = np.mean(np.abs(real-recuperacion[mask]))
print( f"EAM entre recuperacion real y calcaulada: {mae:4f}")



'''
1.3. Analiza las características no disponibles en el conjunto de prueba. 
¿Cuáles son estos parámetros? ¿Cuál es su tipo?
'''
missing_cols = set(df_train.columns) - set(df_test.columns)
print(f"Columna ausentes en el conjunto de prueba:\n", missing_cols)



"1.4. Realiza el preprocesamiento de datos."

df_train.fillna(df_train.mean(numeric_only=True), inplace=True)
df_test.fillna(df_test.mean(numeric_only=True), inplace=True)

print("La cantidad de valores duplicados en el conjunto de entrenamiento: ",df_train.duplicated().sum())
df_test.fillna(df_train.mean(numeric_only=True), inplace=True)

print("La cantidad de valores duplicados en el conjunto de validacion: ",df_train.duplicated().sum())
df_test.fillna(df_test.mean(numeric_only=True), inplace=True)


'''
2.1. Observa cómo cambia la concentración de metales (Au, Ag, Pb) 
en función de la etapa de purificación.
'''

metales = ["au", "ag", "pb"]
etapas = ['rougher.input', 'rougher.output.concentrate', 'final.output.concentrate']

for metal in metales:
    plt.figure(figsize=(10,4))
    for etapa in etapas:
        col = f"{etapa}_{metal}"
        if col in df_train:
            df_train[col].plot(label=col)
    plt.title(f"Concentración de {metal.upper()} por etapa")
    plt.legend()
    plt.show()


'''
2.2. Compara las distribuciones del tamaño de las partículas de la alimentación en el conjunto de entrenamiento y en el conjunto de prueba. 
Si las distribuciones varían significativamente, la evaluación del modelo no será correcta.
'''

plt.figure(figsize=(10,4))
sns.histplot(df_train["rougher.input.feed_size"], label= "Trian", color="blue", kde=True)
sns.histplot(df_test["rougher.input.feed_size"], label= "Test", color="orange", kde=True)
plt.legend()
plt.title("Distribucion del tamaño de las particulas de alimentación")
plt.show()





'''
2.3. Considera las concentraciones totales de todas las sustancias en las diferentes etapas: materia prima, 
concentrado rougher y concentrado final. ¿Observas algún valor anormal en la distribución total? Si es así, 
¿merece la pena eliminar esos valores de ambas muestras? 
Describe los resultados y elimina las anomalías.
'''


def total_concentracion(df, etapa):
    cols = [c for c in df.columns if etapa in c and any(metal in c for metal in metales)]
    return df[cols].sum(axis=1)

for etapa in ['rougher.input', 'rougher.output.concentrate', 'final.output.concentrate']:
    total = total_concentracion(df_train, etapa)
    plt.figure(figsize=(10,4))
    sns.histplot(total, kde=True)
    plt.title(f"Suma de concentraciones en etapa: {etapa}")
    plt.show()

'''
3.1. Escribe una función para calcular el valor final de sMAPE.
'''


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    return np.mean(diff / denominator) * 100




scorer = make_scorer(smape, greater_is_better=True)

model = RandomForestRegressor(n_estimators=100, random_state=12345)

target_r = 'rougher.output.recovery'
target_f = 'final.output.recovery'

excluded_columns = ['date', target_r, target_f]

features = []
for col in  df_train.columns:
    if col not in excluded_columns and "output" not in col:
        features.append(col)

model = RandomForestRegressor(n_estimators=50, random_state=12345)

x_train = df_train[features]
y_rougher = df_train[target_r]
score_rougher = cross_val_score(model, x_train, y_rougher, cv=5, scoring=scorer).mean()

#Entrenamiento
y_final = df_train[target_f]
score_final = cross_val_score(model, x_train, y_final, cv=5, scoring=scorer).mean()

metrica_final = .25 *abs(score_rougher) + .70 *abs(score_final)

print(f"sMAPE de roughter: {abs(score_rougher):2f}")
print(f"sMape final :   {abs(score_final):2f}")
print(f"Metrica final combinada:   {metrica_final:2f}")
#Proyecto Sprint 13
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import sklearn.linear_model
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import sklearn.preprocessing
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math as math 
from IPython.display import display

df = pd.read_csv(r"C:\Users\josep\Downloads\insurance_us (1).csv")
df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})
print(df.sample(n=10))

df["age"] = df["age"].astype("int")
df.info()
print("\n", "Descripción general:")
print(df.describe())

print("\n", "Matriz de correlación:")
print(df.corr())

#Analisis exploratiorio
#g = sns.pairplot(df, kind='hist')
#g.fig.set_size_inches(12, 12)
#plt.show()


#Tarea 1
'''
En el lenguaje de ML, es necesario desarrollar un procedimiento que devuelva los k vecinos más cercanos (objetos) para un objeto dado basándose en la distancia entre los objetos.

Es posible que quieras revisar las siguientes lecciones (capítulo -> lección)

Distancia entre vectores -> Distancia euclidiana
Distancia entre vectores -> Distancia Manhattan
Para resolver la tarea, podemos probar diferentes métricas de distancia.

Escribe una función que devuelva los k vecinos más cercanos para un 
objeto basándose en una métrica de distancia especificada. A la hora de realizar esta tarea no debe tenerse en cuenta el número de prestaciones de seguro recibidas.

Puedes utilizar una implementación ya existente del algoritmo kNN de scikit-learn (consulta el enlace) o tu propia implementación.

Pruébalo para cuatro combinaciones de dos casos

Escalado
los datos no están escalados
los datos se escalan con el escalador MaxAbsScaler
Métricas de distancia
Euclidiana
Manhattan
Responde a estas preguntas:

¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?
¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)?
'''

feature_names = ['gender', 'age', 'income', 'family_members']
def get_knn(df, n, k, metric):
    
    """
    Devuelve los k vecinos más cercanos

    :param df: DataFrame de pandas utilizado para encontrar objetos similares dentro del mismo lugar
    :param n: número de objetos para los que se buscan los vecinos más cercanos
    :param k: número de vecinos más cercanos a devolver
    :param métrica: nombre de la métrica de distancia
    """
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric)# <tu código aquí> 
    nbrs.fit(df[feature_names])
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[n][feature_names]], return_distance=True)
    
    df_res = pd.concat([
        df.iloc[nbrs_indices[0]], 
        pd.DataFrame(nbrs_distances.T, index=nbrs_indices[0], columns=['distance'])
        ], axis=1)
    
    return df_res

#Escalar datos 
feature_names = ['gender', 'age', 'income', 'family_members']

transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[feature_names].to_numpy())

df_scaled = df.copy()
df_scaled.loc[:, feature_names] = transformer_mas.transform(df[feature_names].to_numpy())
print("\n", " Muestra de Datos Escalados:")
print(df_scaled.sample(n=5))

metrics = ['euclidean', 'manhattan']

print(f"\n Datos Sin Escalar:")
for metric in metrics:
    print(f"\n Métrica:{metric}")
    print(get_knn(df, n=0, k=5, metric=metric))

scaler = MaxAbsScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df.drop(columns=['insurance_benefits'])), columns=feature_names)

print(f"\n Datos Escalados:")
for metric in metrics:
    print(f"\n Métrica:{metric}")
    print(get_knn(scaled_df, n=0, k=5, metric=metric))


feature_names = ['gender', 'age', 'income', 'family_members']
metrics = ['euclidean', 'manhattan']

# Escalar
transformer_mas = MaxAbsScaler().fit(df[feature_names])
df_scaled = df.copy()
df_scaled[feature_names] = transformer_mas.transform(df[feature_names].to_numpy()).astype(float)

# Graficar
df['insurance_benefits_received'] =  (df['insurance_benefits'] > 0).astype(int) #<tu código aquí>
print(df['insurance_benefits_received'].value_counts(normalize=True))






#EJERCICIO 2
"""
Con el valor de `insurance_benefits` superior a cero como objetivo, evalúa si el enfoque de clasificación kNN puede funcionar mejor que el modelo dummy.

Instrucciones:

Construye un clasificador basado en KNN y mide su calidad con la métrica F1 para k=1...10 tanto para los datos originales como para los escalados. 
Sería interesante observar cómo *k* puede influir en la métrica de evaluación y si el escalado de los datos provoca alguna diferencia. 
Puedes utilizar una implementación ya existente del algoritmo de clasificación kNN de scikit-learn (consulta el enlace) o tu propia implementación.


Construye un modelo dummy que, en este caso, es simplemente un modelo aleatorio.
Debería devolver "1" con cierta probabilidad. Probemos el modelo con cuatro valores de probabilidad: 0, la probabilidad de pagar cualquier prestación del seguro, 0.5, 1.



La probabilidad de pagar cualquier prestación del seguro puede definirse como:

    número de observaciones con insurance_benefits > 0 / total de observaciones

Divide todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30.
"""
df['insurance_benefits_received'] = (df["insurance_benefits"] > 0).astype("int")
print("\n Prueba del desequilibrio de clase de insurance_benefits:")
print(df['insurance_benefits_received'].value_counts(normalize=True))

features = df.drop(columns=["insurance_benefits", 'insurance_benefits_received'], axis=1)
target = df['insurance_benefits_received']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=.3, random_state=42) 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def eval_classifier(y_true, y_pred):
    
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    print(f'F1: {f1_score:.2f}')
    
# si tienes algún problema con la siguiente línea, reinicia el kernel y ejecuta el cuaderno de nuevo
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='all')
    print('Matriz de confusión')
    print(cm)

print("--"* 60)
print(f"\n KNN con datos originales:")
for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"\nk = {k}")
    eval_classifier(y_test, y_pred)

print("--"* 60)
print(f"\n KNN con datos escalados:")
for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    print(f"\nk = {k}")
    eval_classifier(y_test, y_pred)



# generar la salida de un modelo aleatorio
print("-"* 60)
print("\nModelos Dummy aleatorios:")
def rnd_model_predict(P, size, seed=42):

    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)

for P in [0, df['insurance_benefits_received'].sum() / len(df), 0.5, 1]:

    print(f'La probabilidad: {P:.2f}')
    y_pred_rnd = rnd_model_predict(P, size=len(y_test)) # <tu código aquí> 
        
    eval_classifier(y_test, y_pred_rnd)
    
    print()


#TAREA 3 
"""
Construye tu propia implementación de regresión lineal. Para ello, recuerda cómo está formulada la solución de la tarea de regresión lineal en términos de Álgebra Lineal (LA). Comprueba la RECM tanto para los datos originales como para los escalados. ¿Puedes ver alguna diferencia en la RECM con respecto a estos dos casos?

Denotemos:
- X: matriz de características; cada fila es un caso, cada columna es una característica, la primera columna está formada por unidades.
- y: objetivo (un vector)
- y_hat: objetivo estimado (un vector)
- w: vector de pesos

La tarea de regresión lineal en el lenguaje de matrices puede formularse así:

    y = Xw

El objetivo de entrenamiento es entonces encontrar ese w que minimice la distancia L2 (ECM) entre Xw y y:

    min_w ||Xw - y||^2  o  min_w MSE(Xw, y)

Parece que hay una solución analítica para lo anteriormente expuesto:

    w = (X^T X)^(-1) X^T y

La fórmula anterior puede servir para encontrar los pesos w, y estos últimos pueden utilizarse para calcular los valores predichos:

    y_hat = X_val w
"""
#Divide todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30. Utiliza la métrica RECM para evaluar el modelo.

class MyLinearRegression:
    
    def __init__(self):
        
        self.weights = None
    
    def fit(self, X, y):
        
        # añadir las unidades
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        self.weights = np.linalg.inv(X2.T @ X2) @ X2.T @ y# <tu código aquí>
        return self
    def predict(self, X):
        
        # añadir las unidades
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)# <tu código aquí>
        y_pred = X2 @ self.weights # <tu código aquí>
        
        return y_pred

def eval_regressor(y_true, y_pred):
    
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse:.2f}')
    
    r2_score = math.sqrt(sklearn.metrics.r2_score(y_true, y_pred))
    print(f'R2: {r2_score:.2f}')  


X = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train, y_train)
print("\n pesos de la regresion lineal:",lr.weights)

y_test_pred = lr.predict(X_test)
print("\n Evaluación de la Regresión Lineal:")
eval_regressor(y_test, y_test_pred)

residuals = y_test - y_test_pred
plt.hist(residuals, bins= 30, edgecolor= "black")
plt.title("Histograma de residuos")
plt.xlabel("Error de predicción")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()


#Tarea 4 Ofuscar datos 
"""
Lo mejor es ofuscar los datos multiplicando las características numéricas (recuerda que se pueden ver como la matriz $X$)
por una matriz invertible $P$.

La transformación se expresa como:
                                                X' = X * P

Trata de hacerlo y comprueba cómo quedarán los valores de las características después de la transformación. Por cierto,
la propiedad de invertibilidad es importante aquí, así que asegúrate de que $P$ sea realmente invertible.

Puedes revisar la lección 'Matrices y operaciones matriciales -> Multiplicación de matrices' para recordar la regla
de multiplicación de matrices y su implementación con NumPy.
"""

personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]

#Generar una matriz aleatoria P
rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))

#Comprobar que la matriz P sea invertible
p_inv = np.linalg.inv(P)
print("\n Matriz Inversa:", p_inv)

#¿Puedes adivinar la edad o los ingresos de los clientes después de la transformación?
X_ofuscada = X @ P 
print("\nPrimeras filas de la matriz original (X):\n", X[:5])
print("\nPrimeras filas de la matriz transformada (X'):\n", X_ofuscada[:5])

X_recuperada = X_ofuscada @ p_inv

print("\n Datos originales", X[:5])
print("\n Datos transformados:", X_ofuscada[:5])
print("\n Datos Recuperados", X_recuperada[:5])

### Prueba de que la ofuscación de datos puede funcionar con regresión lineal
"""
Entonces, los datos están ofuscados y ahora tenemos X * P en lugar de tener solo X.
En consecuencia, hay otros pesos w_P como:

    w  = (Xᵀ X)⁻¹ Xᵀ y  
    w_P = [(X P)ᵀ X P]⁻¹ (X P)ᵀ y

¿Cómo se relacionarían w y w_P si simplificáramos la fórmula de w_P anterior?

¿Cuáles serían los valores predichos con w_P?

¿Qué significa esto para la calidad de la regresión lineal si esta se mide mediante la RECM?

Revisa el Apéndice B: Propiedades de las matrices al final del cuaderno.
¡Allí encontrarás fórmulas muy útiles!

Nota: No es necesario escribir código en esta sección, basta con una explicación analítica.
"""


#Respuesta:
#**Respuesta**
"""
Si los datos están ofuscados mediante una matriz P, en lugar de tener solo X, ahora trabajamos con X·P. Esto implica que los nuevos pesos de la regresión, w_P, se calculan con la fórmula:

    w_P = [(X·P)ᵀ · (X·P)]⁻¹ · (X·P)ᵀ · y

Utilizando propiedades de las matrices, como (AB)ᵀ = Bᵀ·Aᵀ, se puede demostrar que:

    w_P = P⁻¹ · w

Esto quiere decir que los nuevos pesos w_P están relacionados con los pesos originales w mediante la inversa de la matriz de transformación P.

En cuanto a las predicciones:

    (X·P) · w_P = X · (P·P⁻¹) · w = X · w

Esto muestra que las predicciones no cambian respecto al modelo original sin ofuscación.

Conclusión:
Aunque los datos hayan sido transformados (ofuscados), las predicciones obtenidas son idénticas y, por tanto, la calidad del modelo medida mediante el RECM (Error Cuadrático Medio) **no se ve afectada**. La regresión lineal sigue funcionando correctamente.
"""
#**Prueba analítica**

### Prueba de regresión lineal con ofuscación de datos
'''
Ahora, probemos que la regresión lineal pueda funcionar, en términos computacionales, con la transformación de ofuscación elegida.
Construye un procedimiento o una clase que ejecute la regresión lineal opcionalmente con la ofuscación. Puedes usar una 
implementación de regresión lineal de scikit-learn o tu propia implementación.
Ejecuta la regresión lineal para los datos originales y los ofuscados, 
compara los valores predichos y los valores de las métricas RMSE y $R^2$. 
¿Hay alguna diferencia?
'''


np.random.seed(42)
n_samples = 100
n_features = 5

X = np.random.rand(n_samples, n_features)
Y = np.dot(X, np.array([1.5, -2.0, 0.7, 3.3, -1.2])) + np.random.randn(n_samples) * 0.5

P= np.random.rand(5,5)
print(P)
P_inv = np.linalg.inv(P)

x_ofus = X @ P

modelo_original = MyLinearRegression().fit(X, Y)
modelo_ofus = MyLinearRegression().fit(x_ofus, Y)

#Predicciones 
y_pred_original =  modelo_original.predict(X)
y_pred_ofuscada = modelo_ofus.predict(x_ofus)

#Desempeño
print("\n Desempeño de predicción original:")
eval_regressor(Y, y_pred_original)
print("\n Desempeño de predicción ofuscada:")
eval_regressor(Y, y_pred_ofuscada)
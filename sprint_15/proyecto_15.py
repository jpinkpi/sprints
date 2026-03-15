#Proyecto_15
import pandas as pd 
import numpy as np
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(r"C:\Users\josep\Downloads\taxi.csv", index_col=[0], parse_dates=[0])
df.sort_index(inplace=True)
df = df.resample("H").sum()
print(df.head())
df.info()






#Análisis
""" Puesto a que estamos limitados a tener que analizar por hora, 
decidi usar una muestra de una semana, para analizar el la tendencia y la estacionalidad."""
df_week = df["2018-04-02 00:00:00" : "2018-04-09 00:00:00"]

decomposed = seasonal_decompose(df_week)

plt.figure(figsize=(8, 8))

plt.subplot(311)
decomposed.trend.plot(ax=plt.gca())
plt.title('Tendencia')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Estacionalidad')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residuales')
plt.tight_layout()  
plt.show() 
#Podemos apreciar que exsite una tendencia signicativa que se relfeja a las 5 am, en donde existen la menor cantidad de viajes. Parece ser que es muy estacionaria.
#Ahora probemos analizando su desviación estandar y moda

df_week = df_week.copy()
df_week["mean"] = df_week["num_orders"].rolling(20).mean()
df_week["std"] = df_week["num_orders"].rolling(20).std()
df_week.plot()
plt.show() 


#Aplicando la diferencia de series temporales
df_week -= df_week.shift() 
df_week["mean"] = df_week["num_orders"].rolling(20).mean()
df_week["std"] = df_week["num_orders"].rolling(20).std()

df_week.plot()
plt.show() 


#Pronostico 
#1 Creación de características

def make_features(data, lags, rolling_window):
    data = data.copy()
    data["hour"] = data.index.hour
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    
    for lag in lags:
        data[f"lag_{lag}"] = data["num_orders"].shift(lag)
    data["rolling_mean"] = data["num_orders"].shift(1).rolling(rolling_window).mean()

    return data

df_feat = make_features(df, lags=[1, 24, 168], rolling_window=24)
df_feat = df_feat.dropna()
train, test = train_test_split(df_feat, shuffle=False, test_size=0.1)

print(train.size)
print(test.size)

features_train = train.drop(["num_orders"],axis=1)
target_train = train["num_orders"]
features_test = test.drop(["num_orders"],axis=1)
target_test = test["num_orders"]

#Solo como checklist
print(features_train.columns)

#Aplicado al modelo de Regresion Lineal
model = LinearRegression()
model.fit(features_train, target_train)

pred_train = model.predict(features_train)
pred_test = model.predict(features_test)

rmse = sqrt(mean_squared_error(target_test, pred_test))

print('RMSE para el conjunto de prueba para la Regresion Lineal:',rmse)

#Modelo Random Forest Regressor

model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)

model.fit(features_train, target_train)
pred_random = model.predict(features_test)
rmse = sqrt(mean_squared_error(target_test, pred_random))
print('RMSE para el conjunto de prueba para el bosque aleatorio:',rmse)


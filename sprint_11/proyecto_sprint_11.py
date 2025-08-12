#Proyecto sprint 11
import pandas as pd 
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


df_data_0 = pd.read_csv(r"C:\Users\josep\Downloads\geo_data_0.csv")
df_data_1 = pd.read_csv(r"C:\Users\josep\Downloads\geo_data_1.csv")
df_data_2 = pd.read_csv(r"C:\Users\josep\Downloads\geo_data_2.csv")

print("\n" + "-" * 60)
print("Región 1:")
print(df_data_0.sample(n=5))
df_data_0.info()


print("\n" + "-" * 60)
print("Región 2:")
print(df_data_1.sample(n=5))
df_data_1.info()

print("\n" + "-" * 60)
print("Región 3:")
print(df_data_2.sample(n=5))
df_data_2.info()

#No pareece ver algun error en suestion, solo pase a revisar la cantidad de valores repetidos, sin embargo, al pareces
#existen psoso registrados con los mimmas id pero no en los valores de las demas columnas, por lo  que lo dejare estatico.


#Contamos los valores duplicados

lista_df = [df_data_0, df_data_1, df_data_2]

df_data_0.name ="Región 1"
df_data_1.name ="Región 2"
df_data_2.name ="Región 3"





for i in lista_df:
    duplicates = i.duplicated().sum()
    print("Numero de valores duplicados", i.name, ":", duplicates)

#Visaulizacion de la distribución:
def grafico(region , region_name="", bins=0):
    datos_numericos = ["f0", "f1", "f2", "product"]

    a = 2 #Numero de filas
    b = 2 #Numero de columnas
    c = 1 #Inicializacion de conteo de plots

    fig = plt.subplots(figsize = (10, 6))

    for i in datos_numericos:
        plt.subplot(a,b,c)
        plt.title(i)
        plt.hist(region[i], bins=bins, linewidth=.8)
        c = c+1
    plt.suptitle(region_name)
    plt.tight_layout()
    plt.show()
    


grafico(df_data_0, "region 1",10)
grafico(df_data_1, "region 2",10)
grafico(df_data_2, "region 3",10)

#Descripcion general de los datos

def descripcion_corr(df, region = ""):

    print("\n" + "-" * 60)
    print(f"Descripción de {region}:")
    print(df.select_dtypes(include='number').drop(columns='id', errors='ignore').describe())
    print(df.select_dtypes(include='number').drop(columns='id', errors='ignore').corr())

descripcion_corr(df_data_0, "region 1")
descripcion_corr(df_data_1, "region 2")
descripcion_corr(df_data_2, "region 3")

print()







# Creacion de funcion para ejercicios posteriores
def preparacion_datos(df):
    df = df.drop("id", axis=1)
    #Estandarizacion
    numeric_data = ["f0", "f1", "f2"]
    scaler = StandardScaler()
    df[numeric_data] = scaler.fit_transform(df[numeric_data])

    features = df.drop('product', axis=1)
    target = df['product']

    #Segmentacion de datos
    features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, train_size=0.75, random_state=54321)
    #Entrenamiento del modelo
    model = LinearRegression()
    model.fit(features_train, target_train)
    #Prediccion 
    predicted_valid = model.predict(features_valid)
    mean = predicted_valid.mean()
    rmse = mean_squared_error(target_valid, predicted_valid)**.5

    print(f"Volumen medio de reservas (validacion):{mean:.4f}     RMSE del modelo: {rmse:.4f}")
    return  features, target, model, mean, rmse, predicted_valid, target_valid






#Entrena y prueba el modelo para cada región en geo_data_0.csv:

print("\n" + "-" * 60)
print("REGION 1 OBSERVACIONES:")
features_0, target_0,model_0, mean_0 , rmse_0, predicted_valid_0,target_valid_0 = preparacion_datos(df_data_0)

print("\n" + "-" * 60)
print("REGION 2 OBSERVACIONES:")
features_1, target_1,model_1, mean_1 , rmse_1 ,predicted_valid_1,target_valid_1= preparacion_datos(df_data_1)

print("\n" + "-" * 60)
print("REGION 3  OBSERVACIONES:")
features_2, target_2,model_2, mean_2 , rmse_2, predicted_valid_2,target_valid_2 = preparacion_datos(df_data_2)


#Prepárate para el cálculo de ganancias:
inversion = 100000000
n_pozos = 200
min_produccion_dls = 500000
min_produccion_uni = 111.1
precio_unidad = 4500
ganancia_por_barril = 4.5


#Presupuesto 
costo_unitario = inversion/n_pozos
print("Presupuesto para el desarrollo de un pozo:", costo_unitario)

punto_eq = costo_unitario/ganancia_por_barril
print("Volumen de reservas para alcanzar el punto de equilibrio de un pozo:", round(punto_eq,2))


def produccion_comparada ( media_region, produccion_minima, region):
    if media_region >= produccion_minima:
        print(f"La produccion media de {region} ({media_region}) supera el minimo requerdio de ({produccion_minima}) para evitar perdidas")
    else:
        print(f"La producción media de {region} ({media_region:4f}) está por debajo de lo requerido de ({produccion_minima}), hay riego de perdidadas")


print("\n" + "-" * 60)
print("Producciones media:")
produccion_comparada(mean_0, min_produccion_uni, "Region 1")
produccion_comparada(mean_1, min_produccion_uni, "Region 2")
produccion_comparada(mean_2, min_produccion_uni, "Region 3")




#Escribe una función para calcular la ganancia de un conjunto de pozos de petróleo seleccionados y modela las predicciones:


def revenue(target, predictions, count):
    predictions_sorted_idx = pd.Series(predictions).sort_values(ascending=False).index
    selected = target.iloc[predictions_sorted_idx[:count]]
    volumen_total = selected.sum()
    ganancia = volumen_total * precio_unidad
    costos = count * costo_unitario

    return ganancia - costos

ganancia_1= revenue(target_valid_0, predicted_valid_0, 200) 
ganancia_2= revenue(target_valid_1, predicted_valid_1, 200) 
ganancia_3= revenue(target_valid_2, predicted_valid_2, 200) 
#DF de los resultados

revenues = pd.DataFrame({
    'region': ['Región 1', 'Región 2', 'Región 3'],
    'revenue_dollars': [round(ganancia_1, 2), round(ganancia_2, 2), round(ganancia_3, 2)]
})

print(revenues)


def bootstrapping(target_valid, predictions, n_iterations=1000, sample_size=500, best_points=200):
    revenues = []

    for i in range(n_iterations):
        # Seleccionamos aleatoriamente 500 puntos (con reemplazo) para simular la exploración
        sampled_indices = pd.Series(range(len(predictions))).sample(n=sample_size, replace=True)
        sample_predictions = pd.Series(predictions).iloc[sampled_indices].reset_index(drop=True)
        sample_target = pd.Series(target_valid).iloc[sampled_indices].reset_index(drop=True)

        # Ordenamos las predicciones y seleccionamos los 200 mejores
        best_indices = sample_predictions.nlargest(best_points).index
        selected_targets = sample_target.iloc[best_indices]

        # Calculamos las ganancias para los 200 mejores puntos seleccionados
        revenue = selected_targets.sum() * 4500 - best_points * 500000
        revenues.append(revenue)

    revenues_df = pd.Series(revenues)
    mean_profit = revenues_df.mean()
    lower_quantile = round(revenues_df.quantile(0.025), 4)
    upper_quantile = round(revenues_df.quantile(0.975), 4)

    confidence_interval = [lower_quantile, upper_quantile]
    loss_risk = (revenues_df < 0).mean() * 100  # Porcentaje de submuestras con pérdidas

    return revenues_df, mean_profit, confidence_interval, loss_risk



revenues_1, mean_profit_1, confidence_interval_1, loss_risk_1 = bootstrapping(target_valid_0, predicted_valid_0)
revenues_2, mean_profit_2, confidence_interval_2, loss_risk_2 = bootstrapping(target_valid_1, predicted_valid_1)
revenues_3, mean_profit_3, confidence_interval_3, loss_risk_3 = bootstrapping(target_valid_2, predicted_valid_2)

# Trazamos histogramas para la distribución de ganancias en cada región
fig, axis = plt.subplots(ncols=3, nrows=1, figsize=(28, 8))
# Construimos un array de una dimensión 1-D
ax = axis.ravel()

# Trazamos histogramas de ganancias para los 200 mejores puntos de las tres regiones
revenues_1.hist(ax=ax[0], bins=15, edgecolor = 'black', linewidth = 0.5)
revenues_2.hist(ax=ax[1], bins=15, edgecolor = 'black', linewidth = 0.5)
revenues_3.hist(ax=ax[2], bins=15, edgecolor = 'black', linewidth = 0.5)

# Establecemos el título para cada eje de nuestro array con set_title
ax[0].set_title('Región 1')
ax[1].set_title('Región 2')
ax[2].set_title('Región 3')
# Colocamos una línea vertical en la posición 0
ax[0].axvline(x=0, color='r', ls='--')
ax[1].axvline(x=0, color='r', ls='--')
ax[2].axvline(x=0, color='r', ls='--')

plt.suptitle('Distribución de ganancias por región', size='xx-large')
plt.show()

# Construimos un dataframe con el beneficio promedio, intervalos de confianza y riesgo de pérdida
revenues_boots = pd.DataFrame({
    'region': ['Región 1', 'Región 2', 'Región 3'],
    'mean_revenue': [round(mean_profit_1, 2),
                     round(mean_profit_2, 2),
                     round(mean_profit_3, 2)],
    'lower_confidence_interval_95%': [round(confidence_interval_1[0],3),
                                      round(confidence_interval_2[0],3),
                                      round(confidence_interval_3[0],3)],
    'upper_confidence_interval_95%': [round(confidence_interval_1[1],3),
                                      round(confidence_interval_2[1],3),
                                      round(confidence_interval_3[1],3)],
    'loss_risk_%' : [loss_risk_1, loss_risk_2, loss_risk_3]

})

print(revenues_boots)


#Proyecto sprint_14
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer




df= pd.read_csv(r"C:\Users\josep\Downloads\car_data.csv")
df.info()
print(df.head(10))
df.describe()
print(df["VehicleType"].sample(10))
print(df["VehicleType"].value_counts())

df["VehicleType"] = df["VehicleType"].astype("category")
df["VehicleType"].fillna("other",inplace=True)



print(df["Gearbox"].sample(10))
print(df["Gearbox"].value_counts())
df["Gearbox"] = df["Gearbox"].astype("category")
df["Gearbox"].fillna("manual",inplace=True)


print(df["FuelType"].value_counts())
df["FuelType"] = df["FuelType"].astype("category")
df["FuelType"].fillna("petrol",inplace=True)

print(df["NotRepaired"].value_counts())
df["NotRepaired"] = df["NotRepaired"].astype("category")
df["NotRepaired"].fillna("no",inplace=True)

#numeric_cols = ["Price", "RegistrationYear", "Power", "Mileage", "RegistrationMonth", "NumberOfPictures", "PostalCode"]
#sample_df = df[numeric_cols].sample(n=500, random_state=42)
#sns.pairplot(sample_df, kind="hist")
#plt.show()


columnas = len(df.columns)
print(f"El numero de columnas son: {columnas}, y son las siguientes:{list(df.columns)}")

#solo para abrir la mente un poco mas y comprender que es lo que estamos trabajando:

variables = ['Price', 'Power', 'Mileage', 'RegistrationYear', 'NumberOfPictures']
sns.pairplot(df[variables], x_vars=['Price'], y_vars=variables[1:])
#plt.show()

#Nos damos cuenta que en realidad no está funcionando ciertas columnas tienen valores demasiado altos 


#Mantener valores realistas para los años de registro
print(df["RegistrationYear"].value_counts())
valid_years = (df["RegistrationYear"] >= 1950) & (df["RegistrationYear"] <= 2025)
df["RegistrationYear"] = df["RegistrationYear"].where(valid_years, pd.NA)
print(df["RegistrationYear"].value_counts())

# Mantener solo valores realistas de potencia para autos de calle
valid_power = (df['Power'] >= 10) & (df['Power'] <= 1000)
df['Power'] = df['Power'].where(valid_power, pd.NA).fillna(df['Power'].mean())
df['Power'] = df['Power'].astype("int")


#variables = ['Price', 'Power', 'Mileage', 'RegistrationYear', 'NumberOfPictures']
variables = ['Price', 'Power', 'Mileage', 'RegistrationYear', 'NumberOfPictures']
sns.pairplot(df[variables], x_vars=['Price'], y_vars=variables[1:])
#plt.show()

df["Model"] = df["Model"].astype("category")
print(df["Model"].value_counts().tail(30))
df.info()

'////////////////////////////////////////////////'
#Entrenamiento de los Modelos

target = df["Price"]
features = df.drop(columns="Price")

#Manaejo de categorías 
cat_features = features.select_dtypes(["category"]).columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=.2, random_state=42)




'/////////////////////////////////////////////////'
#Regresion lineal:
# --- Seleccionar solo columnas numéricas ---
X_train_num = X_train.select_dtypes(include=[np.number]).copy()
X_test_num = X_test.select_dtypes(include=[np.number]).copy()

# --- Imputar valores faltantes ---
imputer = SimpleImputer(strategy="median")

# Ajustar solo con el conjunto de entrenamiento
X_train_num = pd.DataFrame(
    imputer.fit_transform(X_train_num),
    columns=X_train_num.columns,
    index=X_train_num.index
)

# Aplicar al conjunto de prueba con el mismo imputador
X_test_num = pd.DataFrame(
    imputer.transform(X_test_num),
    columns=X_test_num.columns,
    index=X_test_num.index
)


model = LinearRegression()
model.fit(X_train_num, y_train)
y_pred_linear = model.predict(X_test_num)

rmse_linear_reg = sqrt(mean_squared_error(y_test, y_pred_linear))
print("RMSE de la Regresión Linear:", rmse_linear_reg)

'/////////////////////////////////////////////////'


"Modelo de Bosque aleatorio"
#rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
#rf_model.fit(X_train_num, y_train)
#y_pred_rf = rf_model.predict(X_test_num)

#rmse_rf = sqrt(mean_squared_error(y_test, y_pred_rf))
#print()
#print("RMSE del modelo Bosque Aleatorio:", rmse_rf)

''' resultado:
RMSE del modelo Bosque Aleatorio: 2241.622216477884'''

#Modelo Catboost
# 1️⃣ Eliminamos columnas de fecha que no son útiles para el modelo
drop_cols = ['DateCrawled', 'DateCreated', 'LastSeen']
X_train = X_train.drop(columns=drop_cols, errors='ignore')
X_test = X_test.drop(columns=drop_cols, errors='ignore')

# 2️⃣ Detectamos variables categóricas
cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
print("Columnas categóricas detectadas:", cat_features)

# 3️⃣ Rellenamos variables categóricas con "missing"
for col in cat_features:
    if str(X_train[col].dtype) == 'category':
        X_train[col] = X_train[col].cat.add_categories(['missing']).fillna('missing').astype(str)
        X_test[col] = X_test[col].cat.add_categories(['missing']).fillna('missing').astype(str)
    else:
        X_train[col] = X_train[col].fillna('missing').astype(str)
        X_test[col] = X_test[col].fillna('missing').astype(str)

# 4️⃣ Rellenamos variables numéricas con la mediana (para evitar sesgos)
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

for col in num_cols:
    median_value = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_value)
    X_test[col] = X_test[col].fillna(median_value)

# 5️⃣ Verificamos que no queden nulos
assert not X_train.isnull().any().any(), "Hay valores nulos en X_train"
assert not X_test.isnull().any().any(), "Hay valores nulos en X_test"

# 6️⃣ Entrenamiento del modelo CatBoost
cat_model = CatBoostRegressor(
    iterations=500,
    depth=8,
    learning_rate=0.1,
    loss_function="RMSE",
    random_seed=42,
    verbose=100
)

cat_model.fit(X_train, y_train, cat_features=cat_features)

# 7️⃣ Predicción y evaluación
y_pred_cat = cat_model.predict(X_test)
rmse_cat = sqrt(mean_squared_error(y_test, y_pred_cat))

print(f"✅ RMSE de CatBoost: {rmse_cat:.2f}")
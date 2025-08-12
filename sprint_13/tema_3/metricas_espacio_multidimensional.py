import pandas as pd
from scipy.spatial import distance

columns = [
    'dormitorios',
    'superficie total',
    'cocina',
    'superficie habitable',
    'planta',
    'número de plantas',
]
realty = [
    [1, 38.5, 6.9, 18.9, 3, 5],
    [1, 38.0, 8.5, 19.2, 9, 17],
    [1, 34.7, 10.3, 19.8, 1, 9],
    [1, 45.9, 11.1, 17.5, 11, 23],
    [1, 42.4, 10.0, 19.9, 6, 14],
    [1, 46.0, 10.2, 20.5, 3, 12],
    [2, 77.7, 13.2, 39.3, 3, 17],
    [2, 69.8, 11.1, 31.4, 12, 23],
    [2, 78.2, 19.4, 33.2, 4, 9],
    [2, 55.5, 7.8, 29.6, 1, 25],
    [2, 74.3, 16.0, 34.2, 14, 17],
    [2, 78.3, 12.3, 42.6, 23, 23],
    [2, 74.0, 18.1, 49.0, 8, 9],
    [2, 91.4, 20.1, 60.4, 2, 10],
    [3, 85.0, 17.8, 56.1, 14, 14],
    [3, 79.8, 9.8, 44.8, 9, 10],
    [3, 72.0, 10.2, 37.3, 7, 9],
    [3, 95.3, 11.0, 51.5, 15, 23],
    [3, 69.3, 8.5, 39.3, 4, 9],
    [3, 89.8, 11.2, 58.2, 24, 25],
]

df_realty = pd.DataFrame(realty, columns=columns)

vector_first =  df_realty.iloc[3].values # < escribe tu código aquí >
vector_second = df_realty.iloc[11].values# < escribe tu código aquí >

print('Distancia euclidiana:',vector_first )# < escribe tu código aquí >
print('Distancia Manhattan:', vector_second) # < escribe tu código aquí >
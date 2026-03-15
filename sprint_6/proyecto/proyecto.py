import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st 
import seaborn as sns

df = pd.read_csv("/datasets/games.csv")
df.drop_duplicates(inplace=True)
print(df.sample(n=10))
df.info()


#Es necesario cambiar los nombres de las columnas a minúsculas para mantener consistencia y legibilidad¶
df.columns = df.columns.str.lower()

#La columna name tiene algunos valores ausentes. Dado que son pocos, los rellenaré con el valor "no_name".
df["name"]= df["name"].fillna("no_info")

#Los valores de la columna platform deben convertirse en tipo categoría para facilitar su manejo.
df["platform"]= df["platform"].astype("category")

#La columna year_of_release contiene muchos valores ausentes. Rellenaré estos datos con un número irrelevante que me permita identificarlos posteriormente. Además, dado que su tipo actual es flotante, lo convertiré a entero.
df["year_of_release"]=df["year_of_release"].fillna(2018)
print(np.array_equal(df["year_of_release"],df["year_of_release"].astype("int")))#compruebo si es posible la conversión
df["year_of_release"]=df["year_of_release"].astype("int")
#Existen valores ausentes en la columna genre, los cuales rellenaré con el valor "no_info".


df["genre"] = df["genre"].fillna("no_info")
#La columna critic_score será convertida a tipo entero y rellenada con los valores de la media para estandarizar su formato y promedio general, esto es considerando que son bastante los datos que caracen de un a calificación.
df["critic_score"]=df["critic_score"].fillna(df["critic_score"].mean()).astype("int")

#La columna rating presenta una gran cantidad de valores ausentes. Los rellenaré con el valor "no_info" en lugar de excluirlos, ya que descartarlos podría afectar la investigación

df["user_score"]= pd.to_numeric(df["user_score"], errors="coerce").astype("float")
df["user_score"]= df["user_score"] *10
df["user_score"]= df["user_score"].fillna(df["user_score"].mean()).astype("int")

df["total_sales"]= df["na_sales"] + df["eu_sales"] + df["jp_sales"] + df["other_sales"]


#1. Parte 2
#Mira cuántos juegos fueron lanzados en diferentes años. ¿Son significativos los datos de cada período?
df_by_year = df.groupby("year_of_release").size()
print(df_by_year)

df_by_year.plot(kind="bar", xlabel= "Año", ylabel="Cantidad de Juegos", title="Número de Videojuegos respecto años", color= "orange", figsize= [9,7])
plt.show()

#Observa cómo varían las ventas de una plataforma a otra. Elige las plataformas con las mayores ventas totales y construye una distribución basada en los datos de cada año.
#Busca las plataformas que solían ser populares pero que ahora no tienen ventas. ¿Cuánto tardan generalmente las nuevas plataformas en aparecer y las antiguas en desaparecer?

df_by_platform = df.groupby("platform", observed=False)["total_sales"].sum().sort_values(ascending=False).head(6).index
print(df_by_platform)
df_top_platforms = df[df["platform"].isin(df_by_platform)]
print("aqui esta",df_top_platforms)

filtrado = df_top_platforms[["year_of_release","platform", "total_sales"]]
filtrado_grouped= filtrado.groupby(["year_of_release","platform"])["total_sales"].sum().unstack()

selected_platforms = ["PS2", "X360", "Wii", "PS3", "PS", "DS"]
df_final = filtrado_grouped.loc[:, selected_platforms]
print(df_final)

df_final.plot(kind="line", figsize=[9,9], colormap="Paired", title="Distribución de Ventas", xlabel= "Años", ylabel="Millones de Dólares", xlim=[1985,2016])
plt.show()

#1.0.1. Consolas que ya no tienen ventas actuales según el 2016
df_total_platfroms = df.groupby(["year_of_release","platform"])["total_sales"].sum().unstack()
print(df_total_platfroms)
df_2016 = df_total_platfroms.loc[2016]
df_2016_no_sales=(df_2016== 0)

no_sales_platforms = df_2016_no_sales[df_2016_no_sales].index
print(f"Consolas que ya no tienen ventas actuales:{no_sales_platforms}, en total son {no_sales_platforms.size}" )

#1.0.2. Duración Promedio de empresas
df_filtrado = df[df["year_of_release"] != 2018]
df_total_platfroms = df_filtrado.groupby(["year_of_release", "platform"])["total_sales"].sum().reset_index()
df_total_platfroms_agropued = df_total_platfroms[df_total_platfroms["platform"].isin(selected_platforms)].sort_values(by=["platform", "year_of_release"])

print("df_total_platfroms_agropued:",df_total_platfroms_agropued)
ventas_significativas = 12
ventas_no_sig = .1

plataformas = {}
duraciones=[]
for platform in df_total_platfroms_agropued["platform"].unique():
    data = df_total_platfroms_agropued[df_total_platfroms_agropued["platform"] == platform]

    # Años con ventas significativas
    años_buenos = data[data["total_sales"] > ventas_significativas]["year_of_release"]
    # Años con ventas insignificantes
    años_malos = data[(data["total_sales"] > ventas_no_sig) & (data["total_sales"] < ventas_significativas)]["year_of_release"]

    primer_año = años_buenos.min() if not años_buenos.empty else None

    # Se toma el año máximo de ventas no significativas, sino, el primer año donde la plataforma fue descontinuada
    ultimo_año = años_buenos.max() if not años_malos.empty else data["year_of_release"].max()

    # Validor que la fecha de finalización tiene sentido
    if primer_año and ultimo_año and ultimo_año >= primer_año:
        duracion = ultimo_año - primer_año
        duraciones.append(duracion)
    else:
        duracion = None

    plataformas[platform] = {
        "primer_año": primer_año,
        "ultimo_año": ultimo_año,
        "duracion": duracion,
    }



# Imprimo los resultado:
for platform, data in plataformas.items():
    print(f"Plataforma: {platform}, Inicio: {data['primer_año']}, Final: {data['ultimo_año']}, Duración: {data['duracion']} años")

promedio_duracion= sum(duraciones)/len(duraciones)

print(f"El promedio de Duración de las empresas màs signficativas fue de:{promedio_duracion} años.")

#1.0.3. ¿Qué plataformas son líderes en ventas? ¿Cuáles crecen y cuáles se reducen? Elige varias plataformas potencialmente rentables.

ventas=df[(df["year_of_release"]>2011) & (df["year_of_release"] < 2018)]
top= ventas.groupby(["platform", "year_of_release"])["total_sales"].sum().reset_index().sort_values(by=["year_of_release", "total_sales"],ascending=True)
top_filftrado= top[top["total_sales"]>5].reset_index(drop=True)
print(top_filftrado)
markers = "o"
sns.set(style="whitegrid")

for i, platform in enumerate(top_filftrado['platform'].unique()):
    platform_data = top_filftrado[top_filftrado['platform'] == platform]
    plt.plot(platform_data['year_of_release'], platform_data['total_sales'], label=platform, marker=markers[i % len(markers)], linestyle='-', markersize=8)
plt.legend(title="Plataforma", loc="upper left", bbox_to_anchor=(1, 1))
plt.xlabel("Años", fontsize= 14)
plt.ylabel("Ventas en millones USD", fontsize=14)
plt.title("Ventas de las plataformas lideres en los ultimos 4 Años", fontsize=18)
plt.show()

#1.0.4. Predicción de ventas para el 2017
top_filftrado_2= top[top["total_sales"]>10].reset_index(drop=True)
top_filftrado_2.sort_values(by=["year_of_release", "total_sales"])
pivot= top_filftrado_2.pivot(index="platform", columns="year_of_release", values="total_sales").fillna(0)
print(pivot.columns)
pivot[2017]= np.where(pivot[2016]>0, pivot.mean(axis=1),0)
print(pivot)

# Crea un diagrama de caja para las ventas globales de todos los juegos, desglosados por plataforma. ¿Son significativas las diferencias en las ventas? ¿Qué sucede con las ventas promedio en varias plataformas? Describe tus hallazgos.
df_filtrado= df[df["total_sales"]>0]
sns.catplot(data=df_filtrado, x="platform", y="total_sales", kind="box")
sns.color_palette("Paired")
plt.xlabel("plataformas")
plt.title("Ventas globales por plataforma")
plt.ylabel("Millones de Dolares")
sns.set(style="whitegrid")
plt.show()

#1.0.6. Ventas globales según mi top del 2012 -2016

top_filftrado_3= top[top["total_sales"]>0].reset_index(drop=True)
top_filftrado_3.sort_values(by=["year_of_release", "total_sales"])
sns.set_palette("Paired")
for i, platform in enumerate(top_filftrado['platform'].unique()):
    platform_data = top_filftrado_3[top_filftrado_3['platform'] == platform]
    sns.catplot(data=top_filftrado_3, x="platform", y="total_sales", kind="box")
plt.xlabel("lataformas")
plt.title("Ventas globales por plataforma 2012-2016")
plt.ylabel("Millones de Dolares")
plt.show()

print(top_filftrado_3)

#1.0.7. Mira cómo las reseñas de usuarios y profesionales afectan las ventas de una plataforma popular (tu elección). Crea un gráfico de dispersión y calcula la correlación entre las reseñas y las ventas. Saca conclusiones.
plataforma = "X360"

df_plataforma = df[df["platform"]== plataforma]
print(df_plataforma)
sns.scatterplot(x=df_plataforma['user_score'], y=df_plataforma['total_sales'], label='Reseñas de Usuarios', color= "y", alpha=.9)
sns.scatterplot(x=df_plataforma['critic_score'], y=df_plataforma['total_sales'], label='Reseñas Profesionales', color='r', alpha=.9)
plt.xlabel("Puntuaciones 1 al 100")
plt.ylabel("Millones de ventas (USD)")
plt.title("Grafico de Disopersion entre ventas y criticas")
plt.show()
#Correlacion 
corr_usuarios_cri= df_plataforma["user_score"].corr(df_plataforma["total_sales"])
corr_criticos_cri = df_plataforma["critic_score"].corr(df_plataforma["total_sales"])

print(f"""La correlacion para las ventas entre críiticas de usuarios es: {corr_usuarios_cri} 
La correlación para las ventas entre criticas de Criticos es: {corr_criticos_cri} """)

#1.0.8. Echa un vistazo a la distribución general de los juegos por género. ¿Qué se puede decir de los géneros más rentables? ¿Puedes generalizar acerca de los géneros con ventas altas y bajas?
df_genero = df.groupby("genre")["total_sales"].sum()
sns.barplot(df_genero, color="r",errorbar=None)
sns.color_palette("Paired")
plt.title("Ventas historicas por genero")
plt.xlabel("Genero")
plt.ylabel("Millones de Dólares")
plt.show()
print(df_genero)

#2. Parte 3 Crea un perfil de usuario para cada región

ventas_por_region = df.groupby("platform")[["na_sales", "eu_sales", "jp_sales"]].sum()

top_na = ventas_por_region.sort_values(by="na_sales",ascending=False).reset_index().head(5)
print("Top NA:")
print(top_na)
top_na_melt = pd.melt(top_na, id_vars=["platform"], var_name="region", value_name="sales")
sns.barplot(data=top_na_melt,x="platform", y="sales", hue="region", palette="pastel")
plt.title("Top ventas de  NorteAmerica por plataformas, comparado por regiones")
plt.ylabel("Ventas(Millones USD)")
plt.xlabel("plataformas")
plt.show()

#Ventas por plataformas en Europa:
top_eu = ventas_por_region.sort_values(by="eu_sales",ascending=False).reset_index().head(5)
print("Top EU:")
print(top_eu)
top_eu_melt = pd.melt(top_eu, id_vars=["platform"], var_name="region", value_name="sales")
sns.barplot(data=top_na_melt,x="platform", y="sales", hue="region", palette="pastel")
plt.title("Top ventas de  Europa por plataformas, comparado por regiones")
plt.ylabel("Ventas(Millones USD)")
plt.xlabel("plataformas")
plt.show()

#Ventas por plataforma en Japón:
top_jp = ventas_por_region.sort_values(by="jp_sales",ascending=False).reset_index().head(5)
print("Top JP:")
print(top_jp)
top_jp_melt = pd.melt(top_jp, id_vars=["platform"], var_name="region", value_name="sales")
sns.barplot(data=top_na_melt,x="platform", y="sales", hue="region", palette="pastel")
plt.title("Top ventas de  Europa por plataformas, comparado por regiones")
plt.ylabel("Ventas(Millones USD)")
plt.xlabel("plataformas")
plt.show()

#cuota del mercado
ventas_totales = ventas_por_region.sum()
print(ventas_totales)
cuota_mercado = ventas_por_region.div(ventas_totales) * 100
cuota_mayor = cuota_mercado[cuota_mercado>5]
print(cuota_mayor)

#2.0.2. Los cinco géneros por cada región principales. Explica la diferencia

generos_por_region = df.groupby("genre")[["na_sales", "eu_sales", "jp_sales"]].sum()
top_genres_na = generos_por_region.sort_values(by="na_sales",ascending=False).reset_index().head(5)
print("Top Genres NA:")
print(top_genres_na)
top_genres_na_melt = pd.melt(top_genres_na, id_vars=["genre"], var_name="region", value_name="sales")
sns.barplot(data=top_genres_na_melt,x="genre", y="sales", hue="region", palette="pastel")
plt.title("Principales Generos de ventas en NorteAmerica respecto a ventas generales")
plt.ylabel("Ventas(Millones USD)")
plt.xlabel("Generos")
plt.show()

#Ventas por género en Europa
top_genres_eu = generos_por_region.sort_values(by="eu_sales",ascending=False).reset_index().head(5)
print("Top Genres EU:")
print(top_genres_eu)

top_genres_eu_melt = pd.melt(top_genres_eu, id_vars=["genre"], var_name="region", value_name="sales")
sns.barplot(data=top_genres_eu_melt,x="genre", y="sales", hue="region", palette="pastel")
plt.title("Principales Generos de ventas en Europa respecto a ventas generales")
plt.ylabel("Ventas(Millones USD)")
plt.xlabel("Generos")
plt.show()

#Ventas por género en Japón
top_genres_jp = generos_por_region.sort_values(by="jp_sales",ascending=False).reset_index().head(5)
print("Top Genres JP:")
print(top_genres_jp)
top_genres_jp_melt = pd.melt(top_genres_jp, id_vars=["genre"], var_name="region", value_name="sales")
sns.barplot(data=top_genres_jp_melt,x="genre", y="sales", hue="region", palette="dark")
plt.title("Principales Generos de ventas en Japon respecto a ventas generales")
plt.ylabel("Ventas(Millones USD)")
plt.xlabel("Generos")
plt.show()

#Cuota del mercado
generes_totales= generos_por_region.sum()
cuota_del_mercado_genero = generos_por_region.div(generes_totales)*100
print(cuota_del_mercado_genero)

#2.0.3. Investiga si las clasificaciones de ESRB afectan a las ventas en regiones individuales
df_no_rat = df[df["rating"]!= "no_info"]
df_clasificacion= df_no_rat.groupby("rating")[["na_sales", "eu_sales", "jp_sales"]].sum()

top_clafification_na = df_clasificacion.sort_values(by="na_sales",ascending=False).reset_index().head(5)
print("Top Rating NA:")
print(top_clafification_na)
top_clafification_na_melt = pd.melt(top_clafification_na, id_vars=["rating"], var_name="region", value_name="sales")
sns.barplot(data=top_clafification_na_melt, x="rating", y="sales", hue="region", palette="dark")
plt.title("Ventas segun rating de Norte Amercia frente otras regiones")
plt.ylabel("Ventas(Millones USD)")
plt.xlabel("Clasificación")
plt.show()

#2.0.4. En Europa
top_clasification_eu = df_clasificacion.sort_values(by="eu_sales",ascending=False).reset_index().head(5)
print("Top Rating EU:")
print(top_clasification_eu)
top_clasification_eu_melt = pd.melt(top_clasification_eu, id_vars=["rating"], var_name="region", value_name="sales")
sns.barplot(data=top_clafification_na_melt, x="rating", y="sales", hue="region", palette="dark")
plt.title("Ventas segun rating de Europa frente otras regiones")
plt.ylabel("Ventas(Millones USD)")
plt.xlabel("Clasificación")
plt.show()

#En Japón
top_clasification_jp = df_clasificacion.sort_values(by="jp_sales",ascending=False).reset_index().head(5)
print("Top Rating JP:")
print(top_clasification_jp)
top_clasification_jp_melt = pd.melt(top_clasification_jp, id_vars=["rating"], var_name="region", value_name="sales")
sns.barplot(data=top_clafification_na_melt, x="rating", y="sales", hue="region", palette="dark")
plt.title("Ventas segun rating de Japón frente otras regiones")
plt.ylabel("Ventas(Millones USD)")
plt.xlabel("Clasificación")
plt.show()

#Parte 5
# Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas
xbox = df[df["platform"]== "XOne"]
xbox_equal_var = xbox["user_score"]

pc = df[df["platform"]== "PC"]
pc_equal_var = pc["user_score"]

alpha = .05
results= st.ttest_ind(xbox_equal_var, pc_equal_var, equal_var=False)
print('valor p:', results.pvalue) 

if (results.pvalue < alpha):
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")
    
# Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes
accion = df[df["genre"]== "Action"]
deport= df[df["genre"]== "Sports"]

accion_equal_var = accion["user_score"]
deport_equal_var = deport["user_score"]
alpha= .05

results= st.ttest_ind(accion_equal_var, deport_equal_var, equal_var=False)
print('valor p:', results.pvalue) 

if (results.pvalue < alpha):
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")

#Bootstrapping para el análisis de pruebas A/B

'''
Digamos que hemos modificado ligeramente la interfaz de una tienda online. 
Necesitamos comprobar si el importe promedio de compra ha cambiado. 
Podemos utilizar esos datos para responder a la pregunta.

Mientras se realiza la prueba, recopilamos datos sobre el importe de cada compra en el grupo de control y en el grupo experimental. 

Probablemente reconozcas que la comparación de dos grupos por sus medias en alguna variable se parece mucho a la prueba de hipótesis estadísticas. 
El bootstrapping no elimina el concepto de la hipótesis estadística, pero ofrece otra forma de probarlas.

Vamos a formular la hipótesis asociada a la pregunta, pero en términos de comparación de la diferencia entre los promedios, que es

                D=importe promedio de compra en B -importe promedio de compra en A


                Entonces, la hipótesis estadística puede formularse como
                h0 = d=0
                ha = D>0

La hipótesis nula consiste en que no hay diferencia entre los importes promedio de compra en los dos grupos. La hipótesis alternativa dice que el importe promedio de compra es mayor en el grupo experimental. Ahora bien, si calculamos simplemente la diferencia de forma aritmética, es probable que esta varíe ligeramente a causa de la aleatoriedad. Vamos a investigar cuál es la probabilidad de que dicha diferencia se haya obtenido por casualidad (este será nuestro valor p). Si el valor p está por debajo del umbral de significación estadística, podemos rechazar H0 a favor de HA.

La prueba mediante el bootstrapping se realiza de la siguiente manera:

Se calcula la diferencia entre los importes promedio de compra en los dos grupos.
Dada la hipótesis H0 suponemos que los grupos son iguales. Por consiguiente, las observaciones de ambos grupos pueden interpretarse como una sola muestra (extraída de la misma distribución), con lo cual concatenamos las observaciones en una muestra combinada.
Simulamos numerosos experimentos seleccionando dos muestras (una para representar al grupo A y otra para representar al B) de la muestra combinada. Para cada experimento simulado, se calcula la diferencia del importe promedio de compra para esa iteración. Podemos llamarlo Di.
El valor p puede estimarse entonces como la razón entre el número de veces cuando Di no fue inferior a D y el número de experimentos simulados.

Si la disparidad entre los promedios de las muestras originales (D) es simplemente un resultado de la casualidad (aleatoriedad), el proceso aleatorio (combinando las muestras, dibujando nuevas muestras aleatorias y calculando la diferencia entre sus promedios) es probable que no se encuentre con ninguna dificultad para igualar o superar D. Habrá un número sustancial de casos en los que esto ocurra. Sin embargo, si la diferencia es genuina y no solo un resultado de la aleatoriedad (por ejemplo, refleja un verdadero efecto de tratamiento), el bootstrapping puede tener dificultades para producir datos remuestreados con diferencias similares entre las medias, ya que no puede crear nueva información más allá de lo que está presente en los datos originales.

Calculamos la diferencia real de los importes promedio de compra entre los grupos de acuerdo con la fórmula anterior.

D = importe promedio de compra en B - importe promedio de compra en A

A continuación, combinamos las observaciones originales de los grupos A y B.

Determinamos el número de experimentos simulados. Normalmente, este oscila entre cientos y miles. Vamos a utilizar el índice i para referirnos al número de cualquier experimento simulado.

Para llevar a cabo un experimento simulado, se extraen dos muestras aleatorias (con reemplazo) del conjunto combinado de observaciones.

Ai: la muestra del grupo A en el i-ésimo experimento, su tamaño debe ser igual al tamaño de la muestra original del grupo A (según lo observado)
Bi: la muestra del grupo B para el i-ésimo experimento, su tamaño debe ser igual al tamaño de la muestra original del grupo B (según lo observado)

Encuentra la diferencia del importe promedio de compra entre estos dos grupos:

Di = importe promedio de compra en Bi - importe promedio de compra en Ai

Encuentra la estimación del valor p:

p-value = P{Di ≥ D} = (número de experimentos simulados cuando Di ≥ D) / (número de experimentos simulados)

Si el valor p es inferior a un determinado umbral (normalmente, 0,05), podemos rechazar la hipótesis nula y decir que existe una diferencia significativa entre los dos grupos
'''

#EJERCICIO 1
'''
Analiza las dos muestras y comprueba la hipótesis que dice que el importe promedio de compra ha aumentado. 
Guarda la diferencia entre los importes promedio de compra en la variable AB_difference e imprímela en la pantalla. Asigna un nivel de significación del 5 % (.05) a la variable alpha. 
Guarda el valor p en la variable pvalue e imprímelo. 
Imprime el resultado de la prueba de hipótesis.
'''

import pandas as pd
import numpy as np

# datos del grupo de control A
samples_A = pd.Series([
     98.24,  97.77,  95.56,  99.49, 101.4 , 105.35,  95.83,  93.02,
    101.37,  95.66,  98.34, 100.75, 104.93,  97.  ,  95.46, 100.03,
    102.34,  98.23,  97.05,  97.76,  98.63,  98.82,  99.51,  99.31,
     98.58,  96.84,  93.71, 101.38, 100.6 , 103.68, 104.78, 101.51,
    100.89, 102.27,  99.87,  94.83,  95.95, 105.2 ,  97.  ,  95.54,
     98.38,  99.81, 103.34, 101.14, 102.19,  94.77,  94.74,  99.56,
    102.  , 100.95, 102.19, 103.75, 103.65,  95.07, 103.53, 100.42,
     98.09,  94.86, 101.47, 103.07, 100.15, 100.32, 100.89, 101.23,
     95.95, 103.69, 100.09,  96.28,  96.11,  97.63,  99.45, 100.81,
    102.18,  94.92,  98.89, 101.48, 101.29,  94.43, 101.55,  95.85,
    100.16,  97.49, 105.17, 104.83, 101.9 , 100.56, 104.91,  94.17,
    103.48, 100.55, 102.66, 100.62,  96.93, 102.67, 101.27,  98.56,
    102.41, 100.69,  99.67, 100.99])

# datos del grupo experimental B
samples_B = pd.Series([
    101.67, 102.27,  97.01, 103.46, 100.76, 101.19,  99.11,  97.59,
    101.01, 101.45,  94.8 , 101.55,  96.38,  99.03, 102.83,  97.32,
     98.25,  97.17, 101.1 , 102.57, 104.59, 105.63,  98.93, 103.87,
     98.48, 101.14, 102.24,  98.55, 105.61, 100.06,  99.  , 102.53,
    101.56, 102.68, 103.26,  96.62,  99.48, 107.6 ,  99.87, 103.58,
    105.05, 105.69,  94.52,  99.51,  99.81,  99.44,  97.35, 102.97,
     99.77,  99.59, 102.12, 104.29,  98.31,  98.83,  96.83,  99.2 ,
     97.88, 102.34, 102.04,  99.88,  99.69, 103.43, 100.71,  92.71,
     99.99,  99.39,  99.19,  99.29, 100.34, 101.08, 100.29,  93.83,
    103.63,  98.88, 105.36, 101.82, 100.86, 100.75,  99.4 ,  95.37,
    107.96,  97.69, 102.17,  99.41,  98.97,  97.96,  98.31,  97.09,
    103.92, 100.98, 102.76,  98.24,  97.  ,  98.99, 103.54,  99.72,
    101.62, 100.62, 102.79, 104.19])

# diferencia real entre las medias de los grupos
AB_difference =  samples_B.mean() - samples_A.mean()# < escribe tu código aquí >
print("Diferencia entre los importes promedios de compra:", AB_difference)

alpha = .05# <escribe aquí el nivel de significación>
    
state = np.random.RandomState(12345)

bootstrap_samples = 1000
count = 0
for i in range(bootstrap_samples):
    # concatena las muestras
    united_samples = pd.concat([samples_A, samples_B]) # < escribe tu código aquí >

    # crea una submuestra
    subsample = united_samples.sample(frac=1, replace= True,random_state =state) # < escribe tu código aquí >
    
    # divide la submuestra por la mitad
    subsample_A = subsample[:len(samples_A)] # < escribe tu código aquí >
    subsample_B = subsample[len(samples_A):]# < escribe tu código aquí >

    # encuentra la diferencia entre las medias
    bootstrap_difference = subsample_B.mean()- subsample_A.mean()# < escribe tu código aquí >
    
    # si la diferencia no es menor que la diferencia real, añade "1" al contador
    if bootstrap_difference >= AB_difference:
        count += 1

# el valor p es igual al porcentaje de valores excedentes
pvalue = 1. * count / bootstrap_samples
print('p-value =', pvalue)

if pvalue < alpha:
    print("La hipótesis nula se rechaza, a saber, es probable que el importe promedio de las compras aumente")
else:
    print("La hipótesis nula no se rechaza, a saber, es poco probable que el importe medio de las compras aumente")

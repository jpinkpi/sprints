#Cálculo del intervalo de confianza
'''

Cálculo del intervalo de confianza

Vamos a ver cómo construir un intervalo de confianza para la media, basado en la muestra, mediante el teorema central del límite.

Supongamos que tomamos nuestra muestra a partir de una distribución con los siguientes parámetros:

μ = media poblacional  
σ² = varianza poblacional

Denota la media de la muestra:

X̄ = media muestral

El teorema del límite central dice que todas las medias de todas las muestras posibles con un tamaño n se distribuyen normalmente alrededor de la verdadera media poblacional.
"Alrededor" significa que la media de esta distribución de todas las medias muestrales será igual a la verdadera media poblacional.
La varianza será igual a la varianza poblacional dividida entre n (el tamaño de la muestra):

X̄ ~ N(μ, σ² / n)

La desviación estándar de esta distribución se denomina error estándar de la media (SEM, por "Standard Error of the Mean"):

SEM(X̄) = σ / √n

Cuanto mayor sea el tamaño de la muestra, menor será el error estándar. Es decir, todas las medias muestrales estarán más cerca de la media real. Cuanto mayor sea la muestra, más precisa será la estimación.

Vamos a estandarizar esta distribución normal para obtener una con media = 0 y desviación estándar = 1.
Para ello, restamos la media y dividimos entre el error estándar:

Z = (X̄ - μ) / SEM(X̄)  ~  N(0, 1)

A partir de la distribución normal estándar, tomamos el percentil del 5% F(0.05) y el del 95% F(0.95) para obtener el intervalo de confianza del 90%:

P(F(0.05) < (X̄ - μ)/SEM < F(0.95)) = 90%

Reescribiendo la fórmula, despejando μ:

P(X̄ + F(0.05)·SEM < μ < X̄ + F(0.95)·SEM) = 90%

¡Aquí lo tenemos! El intervalo de confianza del 90% para la media real.

Sin embargo, tenemos un problema: para calcular el error estándar, usamos la varianza poblacional σ², pero esta normalmente se desconoce al igual que μ.  
Por tanto, debemos estimarla a partir de la muestra.

Cuando la varianza es desconocida, no podemos usar la distribución normal. En su lugar, usamos la distribución t de Student.  
Reemplazamos los valores críticos F(·) por t(·), obteniendo:

P(X̄ + t(0.05)·SEM < μ < X̄ + t(0.95)·SEM) = 90%

Para simplificar el cálculo, podemos usar la función `scipy.stats.t.interval()` en Python, que toma:

- confidence : nivel de confianza (por ejemplo, 0.90)
- df         : grados de libertad (n - 1)
- loc        : media muestral → sample.mean()
- scale      : error estándar de la media → sample.sem()'''


#EJERCICIO 1
'''
Construye un intervalo de confianza del 95 % para el importe promedio 
de compra en KicksYouCanPayRentWith después de la implementación de la mascota de las zapatillas.
'''

import pandas as pd
from scipy import stats as st

sample = pd.Series([
    439, 518, 452, 505, 493, 470, 498, 442, 497, 
    423, 524, 442, 459, 452, 463, 488, 497, 500,
    476, 501, 456, 425, 438, 435, 516, 453, 505, 
    441, 477, 469, 497, 502, 442, 449, 465, 429,
    442, 472, 466, 431, 490, 475, 447, 435, 482, 
    434, 525, 510, 494, 493, 495, 499, 455, 464,
    509, 432, 476, 438, 512, 423, 428, 499, 492, 
    493, 467, 493, 468, 420, 513, 427])

print('Media:', sample.mean())

confidence_interval = st.t.interval(.95, len(sample)-1,sample.mean(), sample.sem())# < escribe tu código aquí >

print('Intervalo de confianza del 95 %:', confidence_interval)

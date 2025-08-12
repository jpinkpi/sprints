#Comparación de las medias
'''
Hemos realizado la prueba A/B. Ahora, ¿cómo podemos determinar si la nueva funcionalidad es útil? ¿Qué hipótesis debemos aceptar y cuál rechazar?

Analicemos los resultados de la prueba A/B. ¿Qué describe el comportamiento de todos los usuarios? 
¡El valor medio de la métrica! Podría ser el tiempo medio de navegación en el sitio web, el importe medio de las compras o el número medio de usuarios únicos.

Los resultados de las mediciones y los valores medios contienen un elemento aleatorio. 
Por lo tanto, tienen un componente de error aleatorio. No podemos predecir el valor de cada observación con exactitud absoluta, 
pero podemos estimarlo utilizando métodos estadísticos.

Supongamos que nuestra hipótesis nula H₀ dice: la nueva funcionalidad no mejora las métricas. 
Entonces nuestra hipótesis alternativa H₁ será: la nueva funcionalidad mejora las métricas.


En la fase de comprobación de la hipótesis, son posibles dos tipos de errores:

        Error de tipo I: se produce cuando la hipótesis nula es correcta, pero se rechaza (resultado falso positivo. 
        En este caso, la nueva funcionalidad se aprueba y, por lo tanto, es positiva)

        Error de tipo II: se produce cuando la hipótesis nula es incorrecta, pero se acepta (resultado falso negativo)

Para aceptar o rechazar la hipótesis nula, calculemos el nivel de significación, también conocido como valor p (valor de probabilidad). 
El valor p representa la probabilidad de observar los datos, o algo más extremo, suponiendo que la hipótesis nula es verdadera. 
Ayuda a evaluar la probabilidad de cometer un error tipo I (rechazar una hipótesis nula verdadera) en comparación con el nivel de significancia elegido (a).


Ten en cuenta que si el valor p es mayor que el valor de umbral, la hipótesis nula no debería rechazarse. 
Si es menor que el umbral, puede que no valga la pena aceptar la hipótesis nula. Los umbrales generalmente aceptados son del 5 % y del 1 %. 
Pero solo el data scientist toma la decisión final sobre qué umbral podría considerarse suficiente.

Los valores medios se comparan utilizando los métodos de prueba de hipótesis unilateral. 
La hipótesis unilateral se acepta si el valor que se está comprobando es mucho mayor o mucho menor que el de la hipótesis nula.
A nosotros nos interesa la desviación en una sola dirección, que es "mayor que".

Si la distribución de los datos se aproxima a la normalidad (no hay valores atípicos significativos en los datos), 
se utiliza la prueba estándar para comparar las medias. Este método supone una distribución normal de las medias de todas las muestras 
y determina si la diferencia entre los valores comparados es lo suficientemente grande como para rechazar la hipótesis nula.

'''
#EJERCICIO
'''
La tienda online de calzado KicksYouCanPayRentWith agregó a su sitio web una mascota animada de zapatillas deportivas. 
Va saltando por la pantalla mientras el usuario está navegando por los productos.

Observa dos muestras de importe promedio de compra antes y después de la implementación de la mascota. 
Imprime los valores medios anteriores y posteriores.

Comprueba la hipótesis de que el importe promedio de compra ha aumentado. Establece el nivel de significación en 5 %. 
Imprime en la pantalla el valor p y el resultado de la comprobación de la hipótesis.
'''
import pandas as pd
from scipy import stats as st

sample_before = pd.Series([
    436, 397, 433, 412, 367, 353, 440, 375, 414, 
    410, 434, 356, 377, 403, 434, 377, 437, 383,
    388, 412, 350, 392, 354, 362, 392, 441, 371, 
    350, 364, 449, 413, 401, 382, 445, 366, 435,
    442, 413, 386, 390, 350, 364, 418, 369, 369, 
    368, 429, 388, 397, 393, 373, 438, 385, 365,
    447, 408, 379, 411, 358, 368, 442, 366, 431,
    400, 449, 422, 423, 427, 361, 354])

sample_after = pd.Series([
    439, 518, 452, 505, 493, 470, 498, 442, 497, 
    423, 524, 442, 459, 452, 463, 488, 497, 500,
    476, 501, 456, 425, 438, 435, 516, 453, 505, 
    441, 477, 469, 497, 502, 442, 449, 465, 429,
    442, 472, 466, 431, 490, 475, 447, 435, 482, 
    434, 525, 510, 494, 493, 495, 499, 455, 464,
    509, 432, 476, 438, 512, 423, 428, 499, 492, 
    493, 467, 493, 468, 420, 513, 427])

print("La media de antes:", sample_before.mean())# < escribe tu código aquí >)
print("La media de después:", sample_after.mean()) # < escribe tu código aquí >)


# nivel crítico de significación
# la hipótesis se rechaza si el valor p es menor que ese
alpha = .05# < escribe tu código aquí >)

# prueba unilateral (de una cola): el valor p será la mitad
results= st.ttest_ind(sample_before, sample_after)  
pvalue = results.pvalue / 2
print('p-value: ', pvalue)

if pvalue < alpha:
    print(
        "La hipótesis nula se rechaza, a saber, es probable que el importe promedio de las compras aumente"
    )
else:
    print(
        "La hipótesis nula no se rechaza, a saber, es poco probable que el importe medio de las compras aumente"
    )
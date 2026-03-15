
#Metodos Iterativos 
'''
¿Es posible entrenar modelos de regresión lineal más rápido? Los métodos iterativos pueden ayudar al respecto.
La siguiente fórmula se emplea como método directo para entrenar modelos de regresión lineal:
Los métodos directos ayudan a encontrar una solución precisa utilizando una fórmula o un algoritmo determinado. Su complejidad computacional es independiente de los datos. Incluso si todas las respuestas y pesos de la muestra son iguales a cero, el algoritmo seguirá realizando todas las multiplicaciones de la matriz para encontrar la solución.
Otro enfoque para entrenar modelos de regresión lineal es el uso de métodos iterativos o algoritmos iterativos
Sin embargo, estos no te darán una solución precisa, sino solo una aproximada. El algoritmo realiza iteraciones similares repetidamente y la solución se vuelve más precisa con cada paso. 
En el caso de que no se necesite una gran precisión, bastará con unas pocas iteraciones.
La complejidad computacional de los métodos iterativos depende del número de pasos realizados, que puede verse afectado por la cantidad de datos.

'''


#Metodo de biscección 

'''
Vamos a encontrar la solución de la ecuación f(x) = 0 mediante un método iterativo. Vamos a definir f(x) como una función continua. 
Esto significa que su gráfico se puede trazar sin levantar el lápiz del papel.
El método de bisección nos ayudará a resolver nuestra ecuación. 
Este toma una función continua y el segmento [a, b] como entrada. Los valores f(a) y f(b) tienen signos diferentes.
Cuando se cumplen estas dos condiciones:

la función es continua;
los valores de los extremos del segmento tienen signos diferentes,
entonces la raíz de la ecuación se encuentra en algún punto del segmento dado.

En cada iteración, el método de bisección:

Comprueba si algún valor f(a) o f(b) es igual a cero. Si lo es, ya tenemos la solución.
Encuentra el centro del segmento c = (a + b) / 2
Compara el signo de f(c) con los signos de f(a) y f(b):
Si f(c) y f(a) tienen signos diferentes, la raíz se encuentra en el segmento [a, c]. El algoritmo analizará este segmento en su siguiente iteración.
Si f(c) y f(b) tienen signos diferentes, la raíz se encuentra en el segmento [b, c]. El algoritmo analizará este segmento en su siguiente iteración.
Los signos de f(a) y f(b) son diferentes, por lo que no hay más opciones.

La exactitud de la solución se suele elegir de antemano, por ejemplo,  e (margen de error) = 0.000001. 
En cada iteración, el segmento con la raíz se divide entre 2. Una vez que se alcance una longitud de segmento inferior a e, el algoritmo podrá detenerse. Esta condición se denomina criterio de parada.

'''


#Ejercicio 

'''
Termina de escribir el código de la función bisect(). Esta función resolverá la ecuación utilizando el método de bisección, que es uno de los métodos iterativos. La función toma:

function — una función con los valores cero del objetivo. En Python, las funciones se pueden pasar como argumentos. Así es como se llama: function(x)
left, right — extremos izquierdo y derecho del segmento
error — magnitud de error aceptable (la exactitud del algoritmo depende de ella)
La prueba del método para dos funciones ya está en el precódigo.
'''



import math

def bisect(function, left, right, error):
    # El bucle while repite el código mientras se cumpla el criterio.
    # Añadimos el criterio de parada.
    while right - left > error:

        # asegúrate de que no haya ceros
        if function(left) == 0:
            return left
        if function(right) == 0:
            return right

        # < escribe tu código aquí >
        
        # biseca el segmento y encuentra el nuevo
        middle = (left + right)/ 2 # < escribe tu código aquí >
        if  function(left) * function(middle) < 0 :# < escribe tu código aquí 
            right = middle
        else:
            left = middle
        # < escribe tu código aquí >
    return left


def f1(x):
    return x ** 3 - x ** 2 - 2 * x


def f2(x):
    return (x + 1) * math.log10(x) - x ** 0.75


print(bisect(f1, 1, 4, 0.000001))
print(bisect(f2, 1, 4, 0.000001))


import pandas as pd
import matplotlib.pyplot as plt

# Datos del ejemplo 15
data = {
    'Plazo': ['1 mes', '3 meses', '6 meses', '12 meses'],
    'Cetes (%)': [10.5, 10.7, 11.0, 11.3],
    'T-bills (%)': [5.2, 5.4, 5.6, 5.8],
    'Paridad cumplida': [True, True, True, True]  # para anotaciones
}

df = pd.DataFrame(data)

# Crear gráfico de barras
fig, ax = plt.subplots(figsize=(8,5))

bar_width = 0.35
index = range(len(df))

# Barras para Cetes y T-bills
ax.bar(index, df['Cetes (%)'], bar_width, label='Cetes (MXN)', color='skyblue')
ax.bar([i + bar_width for i in index], df['T-bills (%)'], bar_width, label='T-bills (USD)', color='salmon')

# Configuración de ejes
ax.set_xlabel('Plazo')
ax.set_ylabel('Tasa de interés (%)')
ax.set_title('Comparación de tasas y cumplimiento de paridad')
ax.set_xticks([i + bar_width/2 for i in index])
ax.set_xticklabels(df['Plazo'])
ax.legend()

# Anotaciones sobre paridad
for i, paridad in enumerate(df['Paridad cumplida']):
    ax.text(i + bar_width/2, max(df['Cetes (%)'][i], df['T-bills (%)'][i]) + 0.3, 
            '✔' if paridad else '✖', ha='center', fontsize=12, color='green' if paridad else 'red')

plt.tight_layout()
plt.show()
#Regresion Logistica
'''   
La regresión lineal es una red neuronal con una sola neurona. Se puede decir lo mismo sobre la regresión logística. Entonces, ¿cuál es la diferencia? Vamos a averiguarlo.

Si las observaciones tienen solo dos clases, la diferencia entre la regresión lineal y la regresión logística es casi imperceptible. Necesitamos agregar un elemento adicional.

Así es como se ve la regresión lineal:



Y aquí hay regresión logística:



El último diagrama tiene una función sigmoide, o función logística, como función de activación, 
que toma cualquier número real como entrada y devuelve un número en el rango de 0 (sin activación) a 1 (activación).



En la fórmula, E es el número de Euler, que es aproximadamente igual a 2.718281828.

Este número en el rango de 0 a 1 se puede interpretar como una predicción de una red neuronal sobre si la observación pertenece a la clase positiva o a la clase negativa.


Si la suma de los productos de los valores de las entradas y los pesos (z) es muy grande, entonces, en la salida sigmoide, obtenemos un número cercano a la unidad:



Pero si, por el contrario, la suma es un número negativo grande, entonces la función devuelve un número cercano a cero:


La función de pérdida varía dependiendo del tipo de red neuronal. El ECM se usa en tareas de regresión, 
mientras que la entropía cruzada binaria (BCE) es la adecuada para una clasificación binaria. 
No podemos usar la métrica de exactitud porque no tiene un producto, lo cual hace que sea imposible trabajar para DGE.


La BCE se calcula de la siguiente manera:

En la fórmula, p es la probabilidad de obtener una respuesta correcta. La base del logaritmo no importa porque el cambio de la base 
es la multiplicación de la función de pérdida por la constante, que no cambia el mínimo.


Si el objetivo = 1, entonces la probabilidad de respuesta correcta es:
Si el objetivo = 0, entonces p es:



Para comprender mejor la función BCE, observa su gráfico:


 

Si la probabilidad de respuesta correcta p es aproximadamente igual a la unidad, 
entonces -log(p) es un número positivo cercano a cero. Por lo tanto, el error es pequeño.

Si la probabilidad de respuesta correcta p≈0, entonces -log(p) es un número positivo grande. 
Por lo tanto, el error también es grande.
'''
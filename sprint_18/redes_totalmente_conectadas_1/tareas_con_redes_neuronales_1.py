#Tareas con redes neuronales

'''

Ya conoces los conceptos básicos de redes neuronales. Ahora vamos a ver cómo se construyen y se entrenan.
La regresión lineal, el bosque aleatorio y la potenciación del gradiente son buenos para los datos tabulares 
(las filas representan observaciones y las columnas almacenan características). 
Pero ¿qué pasa si los datos son un conjunto de fotos de personas y tenemos que determinar la edad de cada persona con base en la foto? 
¿Cuáles son las observaciones y las características en ese caso?

Supongamos que la observación es una foto y cada pixel es una característica de esa observación.

Aquí hay un ejemplo: la imagen de la izquierda es una imagen en escala de grises de baja resolución. 
La del medio es la misma imagen, pero tiene indicado el valor de brillo de cada pixel, que va de 0 (negro) a 255 (blanco).
Estos valores son el conjunto de posibles características. 
La imagen de la derecha contiene solo los números del rango sin la imagen.

Vamos a convertir los valores de píxel en un vector para obtener nuestras características:

[255, 255, 255, 255, 237, 217, 239, 255, 255, 255, 255, 255, 255, 255, 255, 190, 75, 29, 29, 30, 81, 198, 255, 255, 255, 255, 255, 147, 30, 29, 29, 29, 29, 29, 31, 160, 255, 255, 255, 185, 29, 29, 29, 29, 29, 29, 29, 29, 31, 198, 255, 255, 61, 29, 29, 29, 29, 29, 29, 29, 29, 29, 74, 255, 255, 108, 121, 121, 121, 121, 121, 121, 121, 121, 121, 102, 219, 255, 250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 238, 107, 168, 255, 238, 153, 150, 152, 244, 201, 152, 150, 178, 253, 103, 144, 248, 243, 121, 108, 114, 225, 184, 130, 112, 154, 235, 103, 62, 197, 255, 227, 168, 231, 149, 230, 196, 179, 251, 183, 29, 29, 105, 255, 255, 219, 195, 191, 184, 195, 235, 255, 91, 29, 29, 30, 187, 255, 234, 218, 218, 218, 218, 243, 174, 29, 29, 29, 29, 38, 180, 255, 255, 255, 255, 255, 169, 35, 29, 29, 29, 29, 29, 29, 82, 153, 174, 150, 76, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29]
Si el tamaño de cada imagen del conjunto de datos es de 1920x1080 pixeles, 
entonces cada imagen se describe mediante 207 360 características (1920 multiplicado por 1080).
Los algoritmos clásicos como la potenciación del gradiente no pueden administrar el entrenamiento con tantas funciones.

Pensemos en otro ejemplo en el que el conjunto de datos conste de descripciones de texto de los productos de una tienda online y 
tengamos que determinar la categoría de cada artículo.

Podemos usar la técnica de la bolsa de palabras para convertir texto en números. 
Primero, creamos una nueva columna para cada palabra. Si la palabra se encuentra en la descripción del artículo, ponemos un 1 en la columna. En caso contrario, ponemos un 0. Si la descripción tiene entre 10 y 20 palabras, podemos entrenar la regresión logística utilizando el conjunto de datos modificado, pero cuando tenemos cientos de palabras, este método de codificación no funciona. Además, es importante tener en cuenta que la bolsa de palabras no considera el orden de las palabras y eso puede alterar el significado del texto. Si usamos los n-gramas, la cantidad de características será aún mayor, por lo que los métodos clásicos no funcionan aquí.

Veamos qué tienen en común las imágenes y los textos:

1 Ambos tienen información redundante. Por ejemplo, los pixeles de fondo no son 
tan importantes como los que muestran las arrugas o las canas a la hora de determinar la edad de una persona en la fotografía. 
Del mismo modo, las preposiciones y las conjunciones no tienen tanto significado en un texto como los verbos y los sustantivos. 
Las redes neuronales ayudan a procesar una gran cantidad de características.


2.Las características vecinas están relacionadas entre sí. 
A menudo, los pixeles adyacentes pertenecen al mismo objeto y las palabras vecinas en una oración tienen un significado relacionado. 
Mezclar columnas de datos tabulares no cambia su significado, mientras que mezclar pixeles en una imagen producirá ruido. 
Las redes neuronales pueden tener en cuenta el orden de las características.
'''

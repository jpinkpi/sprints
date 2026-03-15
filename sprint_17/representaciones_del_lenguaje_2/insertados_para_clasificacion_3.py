#Insertados para Clasificación
'''
Ahora vamos a descubrir cómo la representación vectorial puede ayudar a resolver tareas de clasificación y regresión.
Digamos que hay un corpus de texto que necesita ser clasificado. En ese caso, nuestro modelo constará de dos bloques:

Modelos para convertir palabras en vectores: las palabras se convierten en vectores numéricos.
Modelos de clasificación: se utilizan vectores como características.

Vamos a repasar los detalles.

1.Antes de pasar a la vectorización de palabras, necesitaremos realizar un preprocesamiento:
Cada texto está tokenizado (descompuesto en palabras).
Luego se lematizan las palabras (reducidas a su forma raíz). Sin embargo, los modelos más complejos, como BERT, no requieren este paso porque entienden las formas de las palabras.
El texto se limpia de palabras vacías o caracteres innecesarios.
Para algunos algoritmos (por ejemplo, BERT), se agregan tokens especiales para marcar el comienzo y el final de las oraciones.

2.Cada texto adquiere su propia lista de tokens después del preprocesamiento.

3.Luego, los tokens se pasan al modelo, que los vectoriza mediante el uso de un vocabulario de tokens precompilado. En la salida obtenemos vectores de longitud predeterminada formados para cada texto.

4.El paso final es pasar las características (vectores) al modelo. Luego, el modelo predice la tonalidad del texto: "0" — negativo o "1" — positivo.

'''


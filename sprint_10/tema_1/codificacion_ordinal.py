#Codificación ordinal

'''
En la lección anterior mencionamos que la codificación ordinal difiere de la de etiquetas. ¿Cuál es la diferencia?
Las variables categóricas que cubrimos hasta ahora no tenían ningún tipo de orden natural implícito. Para categorías como "Plan de valor", "Plan de plata" y "Plan de cancelación", no había una forma obvia de decidir qué categoría se etiquetaba como 0, 1 o 2. 
Así que simplemente asignamos estas etiquetas de forma arbitraria.
Sin embargo, hacerlo no siempre es una buena idea. 
Imagina una variable que contiene descripciones de evaluación de calidad, como "excelente", "buena" y "mala". 
O quizás una variable que contenga descripciones de temperatura, como "caliente", "tibia" y "fría". 
Este tipo de variables categóricas tienen un orden natural, por lo que asignarles etiquetas de forma arbitraria sería un error. 
Si codificas "fría" con 0, "caliente" con 1 y "tibia" con 2, entonces el algoritmo funcionará suponiendo que "caliente" es una cualidad entre "fría" y "tibia", 
en lugar de una cualidad aún más cálida que "tibia".

Este tipo de variable categórica se denomina variable ordinal,
 a diferencia de una variable nominal (una variable de categorías sin orden). 
 La codificación ordinal es una codificación de una variable ordinal con etiquetas numéricas dispuestas en un orden natural específico, 
 generalmente realizada mediante enumeración manual de etiquetas.

Es técnicamente posible implementar la codificación ordinal mediante la clase OrdinalEncoder en sklearn. 
Para hacerlo, debes especificar el parámetro categories. Sin embargo, 
la clase OrdinalEncoder de sklearn no es fácil de usar cuando se trata de codificación ordinal, 
por lo que sugerimos usar una clase OrdinalEncoder alternativa de la librería category_encoders. 
Su parámetro mapping está mejor documentado y es mucho más intuitivo de usar. Otra alternativa es simplemente implementar el mapeo por medio de pandas. 
Ya que estamos usando pandas de todos modos, este es el método que recomendamos:
'''
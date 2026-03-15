#WORD2VEC

'''

Hablemos de word2vec (palabra a vector), un método popular para construir representaciones de lenguaje.
Entonces, ¿cómo funciona word2vec? Recordarás que en la lección anterior, las palabras "pinzón" y "frailecillo" se consideraron similares porque las dos se pueden usar en contexto con picos rojos, 
y las palabras "oso hormiguero" y "perezoso" se pueden usar en enunciados sobre Perú. Es decir, el significado de las palabras depende del contexto.


Word2vec es un método que se utiliza para convertir palabras en vectores (de ahí su nombre) para que las palabras cercanas semánticamente acerquen los vectores entre sí. 
Las librerías que ofrecen este método suelen contener modelos word2vec entrenados previamente, con cada modelo entrenado en un corpus específico de textos.

Por lo tanto, para obtener vectores relevantes para tus textos, debes seleccionar un modelo word2vec que se haya creado para un corpus de textos semánticamente relevante.
Por ejemplo, para convertir noticias en vectores, se debe usar un modelo word2vec entrenado en un corpus de noticias.
Puedes consultar la lista de modelos capaces de convertir palabras en inglés en vectores enviados con spaCy (materiales en inglés) en el enlace https://spacy.io/models/en (materiales en inglés) 
(comprueba los que tengan definida la propiedad "Vectors"). La otra librería popular para implementar el método word2vec es gensim (materiales en inglés), 
y la puedes consultar en el enlace (materiales en inglés). Esto es solo para darte una idea general de cómo se ven los modelos previamente entrenados.


Además de convertir palabras en vectores, los modelos word2vec también se pueden usar para resolver tareas específicas de PNL. 
Por ejemplo, pueden ayudar a predecir si algunas palabras son vecinas o no. Las palabras se consideran vecinas si caen en la misma "ventana" (distancia máxima entre palabras). 
Dado que cada palabra en un par de palabras es una característica, esto significa que hay dos características en cada par. El objetivo es averiguar si las palabras del par son vecinas o no.


Vamos a usar el enunciado sobre el "puffin" (frailecillo) como el conjunto de datos para nuestra tarea:

Red beak of the puffin flashed in the blue sky
Aquí está el mismo texto después de la lematización:

red beak puffin flash in blue sky
La palabra que predecimos en el enunciado es "puffin". Las palabras vecinas son las más cercanas de ambos lados: "red", "beak", "flash" y "in". Por ejemplo, estas cinco palabras forman un cincograma:


Ahora tenemos cuatro pares de vecinas:

puffin    red
puffin    beak
puffin    flash
puffin    in
Si tuviéramos que predecir la palabra "flash", las palabras vecinas serían: "beak," "puffin," "in," and "blue".

Ahora tenemos cuatro nuevos pares de palabras vecinas:

puffin    red
puffin    beak
puffin    flash
puffin    in
flash    beak
flash    puffin
flash    in
flash    blue


Podemos "desplazar la ventana" a través de todo el texto y extraer la lista completa de palabras vecinas.
Otra cosa que puede hacer word2vec es entrenar un modelo para distinguir pares de vecinas verdaderas de las aleatorias. 
Esta tarea es como una tarea de clasificación binaria, donde las características son palabras y el objetivo es la respuesta a la pregunta de si las palabras son vecinas verdaderas o no.

Ya tenemos nuestros ejemplos positivos, pero para obtener los negativos necesitaremos tomar palabras aleatorias del corpus y emparejarlas:

puffin juan
beak bizarre
sky animal
red tell
Ahora tenemos un conjunto de entrenamiento. Aquí, "1" indica que las palabras son verdaderas vecinas, mientras que "0" significa que no son vecinas.
'''

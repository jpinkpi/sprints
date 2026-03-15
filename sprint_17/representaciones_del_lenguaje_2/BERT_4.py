#BERT

'''
Vamos a echar un vistazo al modelo BERT para descubrir cómo ayuda a convertir palabras en vectores.

BERT (Representaciones de Codificador Bidireccional de Transformadores) es un modelo de red neuronal creado para la representación del lenguaje. 
Fue creado por Google para mejorar la relevancia de los resultados de búsqueda y se publicó en 2018 
(el artículo original se encuentra en:https://arxiv.org/abs/1810.04805 (Estos materiales están en inglés)). 
Este algoritmo es capaz de "comprender" el contexto de un texto completo, no solo de frases cortas. 
BERT se usa con frecuencia en el machine learning para convertir textos en vectores. 
Los especialistas suelen utilizar modelos BERT existentes que están previamente entrenados (por Google o, posiblemente, por otros colaboradores) en grandes corpus de texto. 
Los modelos BERT previamente entrenados funcionan para muchos idiomas (104, con exactitud). 
Puedes entrenar tu propio modelo de representación del lenguaje, pero este requerirá muchos recursos computacionales. 


BERT es un paso evolutivo en comparación con word2vec. 
BERT se convirtió rápidamente en la opción popular para los programadores y ha inspirado a los investigadores a crear otros modelos de representación de lenguaje: 
FastText, GloVe (Vectores globales para representación de palabras), ELMO (Insertados del modelo de lenguaje), GPT (Transformador generativo de preentrenamiento). 
Los modelos más precisos actualmente son BERT y GPT.

Al procesar palabras, BERT considera tanto las palabras vecinas inmediatas como las palabras más lejanas. 
Esto permite que BERT produzca vectores precisos con respecto al significado natural de las palabras.

Así es como funciona:

Aquí hay un ejemplo de entrada para el modelo: "The red beak of the puffin [MASK] in the blue [MASK] ", donde MASK representa palabras desconocidas o enmascaradas. 
El modelo tiene que adivinar cuáles son estas palabras enmascaradas.
El modelo aprende a averiguar si las palabras del enunciado están relacionadas. Teníamos enmascaradas las palabras "flashed" y "sky". 
El modelo tiene que comprender que una palabra sigue a la otra. Entonces, si ocultáramos la palabra "crawled" en lugar de "flashed", el modelo no encontraría una conexión.

'''

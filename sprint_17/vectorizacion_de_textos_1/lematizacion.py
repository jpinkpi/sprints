#Lematización
'''
Antes de extraer características, vamos a tener que simplificar el texto. Esto ayudará a que el texto sea más fácil de manejar debido a las variaciones en las formas de las palabras.

Estos son los pasos para el preprocesamiento de texto:

Tokenización: dividir el texto en tokens (frases, palabras y símbolos separados);
Lematización: reducir las palabras a sus formas fundamentales (lema).
Puedes usar estas librerías tanto para la tokenización como para la lematización:

Natural Language Toolkit (NLTK)
spaCy
Hay otras librerías que puedes usar para la tarea (por ejemplo, UDPipe, word2vec), pero NLTK y spaCy son las opciones más populares.

Importa la función de tokenización y crea un objeto de lematización:

'''

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

lemmatizer  = WordNetLemmatizer()


'''
Pasa a la función lemmatize() el texto "All models are wrong, but some are useful" ("Todos los modelos son incorrectos, pero algunos son útiles") como tokens separados:
'''
text = "All models are wrong, but some are useful."

tokens = word_tokenize(text.lower())

lemmas = [lemmatizer.lemmatize(token) for token in tokens]

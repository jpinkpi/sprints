#Creacion de bolsa de Palabras 
'''
En esta lección, vamos a aprender cómo crear una bolsa de palabras y encontrar palabras vacías.
Para convertir un corpus de texto en una bolsa de palabras, usa la clase CountVectorizer() del módulo sklearn.feature_extraction.text.

Importa la clase:
'''
from sklearn.feature_extraction.text import CountVectorizer

#Crea un contador:

count_vect = CountVectorizer()
corpus= "laralara la cuchara"

'''
Pasa el corpus de texto al contador. Llama a la función fit_transform(). 
El contador extrae palabras únicas del corpus y cuenta cuántas veces aparecen en cada texto del corpus. 
El contador no cuenta letras separadas.
'''
# bow = bolsa de palabras
#bow = count_vect.fit_transform(corpus)


'''
Este método devuelve una matriz donde las filas representan textos y las columnas muestran palabras únicas del corpus. 
Los números en sus intersecciones representan cuántas veces aparece una determinada palabra en el texto.

Usemos el corpus (ya lematizado) de la lección anterior:
'''

corpus = [
    'for want of a nail the shoe be lose',
    'for want of a shoe the horse be lose',
    'for want of a horse the rider be lose',
    'for want of a rider the message be lose',
    'for want of a message the battle be lose',
    'for want of a battle the kingdom be lose',
    'and all for the want of a horseshoe nail'
]


#Vamos a crear una bolsa de palabras para la matriz. Utiliza el atributo shape para descubrir el tamaño de la matriz:
#bow.shape

'(7, 16)' # <--- La respuesata al llamar shape


'''
El resultado es 7 textos y 16 palabras únicas.
Aquí está nuestra bolsa de palabras como una matriz:
'''

#print(bow.toarray())

'''
[[0 0 0 1 0 0 0 1 0 1 1 0 1 1 1 1]
 [0 0 0 1 1 0 0 1 0 0 1 0 1 1 1 1]
 [0 0 0 1 1 0 0 1 0 0 1 1 0 1 1 1]
 [0 0 0 1 0 0 0 1 1 0 1 1 0 1 1 1]
 [0 0 1 1 0 0 0 1 1 0 1 0 0 1 1 1]
 [0 0 1 1 0 0 1 1 0 0 1 0 0 1 1 1]
 [1 1 0 1 0 1 0 0 0 1 1 0 0 1 1 0]]         #<-------- La salida

'''
#La lista de palabras únicas en la bolsa se llama vocabulario; 
# se almacena en el contador y se puede acceder a ella llamando al método get_feature_names():

count_vect.get_feature_names()


#Aquí está el vocabulario para nuestro ejemplo:


['all',
 'and',
 'battle',
 'for',
 'horse',
 'horseshoe',
 'kingdom',
 'lost',
 'message',
 'nail',
 'of',
 'rider',
 'shoe',
 'the',
 'want',
 'was']



'''
CountVectorizer() también se usa para cálculos de n-gramas. 
Especifica el tamaño del n-grama con el argumento ngram_range para que cuente las frases. 
Necesitarás dos números enteros para establecer el tamaño mínimo y máximo de n-gramas.
Si solo necesitamos bigramas (frases de dos palabras), entonces lo haremos de esta manera:
'''
count_vect = CountVectorizer(ngram_range=(2, 2))

'''
El contador funciona de la misma forma con frases y con palabras.
Dado que un corpus grande representa una bolsa de palabras más grande, 
algunas de las palabras pueden mezclarse y terminar causando más confusión que claridad. 
Para ayudar con esto, por lo general, puedes eliminar las conjunciones y las preposiciones sin perder el significado de la oración. 
Si tienes una bolsa de palabras más pequeña y limpia, encontrarás más fácilmente las palabras más importantes para la clasificación del texto.

Para asegurarte de obtener una bolsa de palabras más limpia, encuentra las palabras vacías (palabras que no significan nada por sí solas). 
Hay muchas de ellas, y son diferentes para cada idioma: en inglés, por ejemplo, hay artículos ("a", "the") y preposiciones ("in", "for").
Echemos un vistazo al paquete stopwords del módulo nltk.corpus:
'''
from nltk.corpus import stopwords

#Deberás descargar el paquete una vez para que funcione:
import nltk
nltk.download('stopwords')

#Llama a la función stopwords.words() y utiliza 'english' como un argumento para obtener un conjunto de palabras vacías para inglés:
stop_words = set(stopwords.words('english'))

#Pasa la lista de palabras vacías al CountVectorizer() cuando crees el contador:
count_vect = CountVectorizer(stop_words=stop_words)

#Ahora el contador sabe qué palabras se deben excluir de la bolsa de palabras. 
# Si lo ejecutas para el ejemplo anterior, la versión final de la bolsa de palabras se verá así:

['battle',
 'horse',
 'horseshoe',
 'kingdom',
 'lose',
 'message',
 'nail',
 'rider',
 'shoe',
 'want']


#Ejercicios 

#1
'''
1.

imdb_reviews_small_lemm.tsv contiene el conjunto de datos imdb_reviews_small.tsv 
al que agregamos la columna review_lemm con reseñas limpias y lematizadas.
Crea dos bolsas de palabras para el corpus de reseñas: con y sin palabras vacías. Imprime sus tamaños (en precódigo).
'''
try: 
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import stopwords
    # < escribe tu código aquí >

    data = pd.read_csv('/datasets/imdb_reviews_small_lemm.tsv', sep='\t')
    corpus = data['review_lemm']
    count_vect = CountVectorizer()
    bow = count_vect.fit_transform(corpus)


    # crea una bolsa de palabras con la comprobación de las palabras vacías
    # < escribe tu código aquí >

    print('El tamaño de BoW con palabras vacías:', bow.shape)
    stop_words = set(stopwords.words('english'))
    count_vect= CountVectorizer(stop_words=stop_words)
    bow= count_vect.fit_transform(corpus)
    # crea una bolsa de palabras sin comprobar las palabras vacías
    # < escribe tu código aquí >

    print('El tamaño de BoW sin palabras vacías:', bow.shape)

except:print("PRUEBA")


#EJERCICIO 2

'''
2.
Crea un contador de n-gramas para el corpus de reseñas. 
Cada frase debe tener dos palabras. Imprime el tamaño del n-grama (en precódigo).
'''

try:
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    # importa CountVectorizer

    data = pd.read_csv('/datasets/imdb_reviews_small_lemm.tsv', sep='\t')
    corpus = data['review_lemm']
    count_vect = CountVectorizer(ngram_range=(2,2))
    n_gram = count_vect.fit_transform(corpus)

    # crea un n-grama con n=2 y guárdalo en la variable n_gram

    # < escribe tu código aquí >

    print('El tamaño del bigrama:', n_gram.shape)
except:print("PRUEBA")
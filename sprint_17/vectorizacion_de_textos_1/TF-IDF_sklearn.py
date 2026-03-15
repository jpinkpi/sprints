'''
Docstring for TF-IDF_sklearn

Vamos a calcular la TF-IDF para un corpus de texto.

Puedes calcularla mediante la librería sklearn. 
La clase TfidfVectorizer() se encuentra en el módulo sklearn.feature_extraction.text. 
Impórtala como se muestra a continuación:
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

corpus = [
    'for want of a nail the shoe be lose',
    'for want of a shoe the horse be lose',
    'for want of a horse the rider be lose',
    'for want of a rider the message be lose',
    'for want of a message the battle be lose',
    'for want of a battle the kingdom be lose',
    'and all for the want of a horseshoe nail'
]


#Crea un contador y define palabras vacías, tal como lo hicimos con CountVectorizer():

stop_words = set(stopwords.words('spanish'))
count_tf_idf = TfidfVectorizer(stop_words=stop_words)

#Llama a la función fit_transform() para calcular la TF-IDF para el corpus de texto:

tf_idf = count_tf_idf.fit_transform(corpus)

'''
Podemos calcular los n-gramas al pasar el argumento ngram_range a TfidfVectorizer().

Si los datos se dividen en conjuntos de entrenamiento y prueba, llama a la función fit() solo para el conjunto de entrenamiento. 
De lo contrario, la prueba estará sesgada, porque el modelo tomará en cuenta las frecuencias de las palabras del conjunto de prueba.
'''

#EJERCICIO 1
'''
Crea una matriz con valores TF-IDF para el corpus de reseñas. Guárdala en la variable tf_idf. Imprime el tamaño de la matriz (en precódigo).

'''
try:
    import pandas as pd
    from nltk.corpus import stopwords as nltk_stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    # import TfidfVectorizer
    #  < escribe tu código aquí >

    data = pd.read_csv('/datasets/imdb_reviews_small_lemm.tsv', sep='\t')
    corpus = data['review_lemm']

    stop_words = set(nltk_stopwords.words('english'))
    count_tf_idf = TfidfVectorizer(stop_words=stop_words)
    tf_idf = count_tf_idf.fit_transform(corpus) #  < escribe tu código aquí >

    print('El tamaño de la matriz TF-IDF:', tf_idf.shape)

except:print("Prueba")



#Expresiones Regulares 
'''
Una expresión regular es un instrumento para encontrar patrones complejos en los textos.
Puedes manipular a tu gusto las ocurrencias de un patrón (extraer, reemplazar, etc.). 
¡Las expresiones regulares son herramientas poderosas que se usan en casi todos los lugares donde aparece un texto!

Las expresiones regulares encuentran secuencias de caracteres, palabras y números mediante el reconocimiento de patrones. 
Por ejemplo, si necesitáramos encontrar todas las fechas escritas en el formato DD.MM.AAAA, 
entonces tendríamos que usar el siguiente patrón: dos números, un punto, dos números, un punto, cuatro números.

El patrón de una dirección de correo electrónico sería: una cadena alfanumérica, @, 
una cadena alfanumérica, punto, una cadena alfanumérica.

Python tiene un módulo integrado para trabajar con expresiones regulares,  re:

'''
import re

'''Echa un vistazo a la función re.sub().
Esta encuentra todas las partes del texto que coinciden con el patrón dado 
y luego las sustituye con el texto elegido.
'''

# patrón
# sustitución: con qué debe sustituirse cada coincidencia de patrón
# texto: el texto que la función escanea en busca de coincidencias de patrón


"re.sub(pattern, substitution, text)"

'''
Las expresiones regulares tienen su propia sintaxis que puede describir varias combinaciones de cadenas. 
Una simple expresión regular "a.b" coincidirá con cualquier cadena de tres caracteres que comience con "a" y termine con "b". 
El punto indica que cualquier carácter puede aparecer en la segunda posición.

Las expresiones regulares suelen utilizar el carácter de barra invertida ('\') como parte de su sintaxis. 
Dado que esto puede causar un problema de interpretación debido a los caracteres de escape, las expresiones regulares se definen usando cadenas sin formato.

Aquí hay un ejemplo rápido de la diferencia entre una cadena normal y una cadena sin formato. 
Considera que definimos una cadena sin formato escribiendo r antes de la cadena:

'''
print('¡Hola!\n')
print(r'¡Hola!\n')

#Ahora, veamos el siguiente texto de una reseña:

text = """
I liked this show from the first episode I saw, which was the "Rhapsody in Blue" episode (for those that don't know what that is, the Zan going insane and becoming pau lvl 10 ep). Best visuals and special effects I've seen on a television series, nothing like it anywhere.
"""

'''
Como parte del paso de preprocesamiento, debemos eliminar todos los caracteres excepto 
las letras, los apóstrofos y los espacios, así que vamos a escribir una expresión regular para encontrarlos.
Todas las letras que coinciden con el patrón se enlistan entre corchetes, sin espacios, y se pueden colocar en cualquier orden. 
Encontremos letras de la a a la z. Si asumimos que pueden estar tanto en minúsculas como en mayúsculas, 
entonces el código debería escribirse de la siguiente manera:
'''

# un rango de letras se indica con un guión:
# a-z = abcdefghijklmnopqrstuvwxyz
pattern = r"[a-zA-Z]"


#Si también queremos encontrar apóstrofos, podemos agregar uno a la expresión regular:

pattern = r"[a-zA-Z']"

'''Si llamamos a re.sub(pattern, ' ', text), se sustituirán todas las letras y apóstrofos, 
pero necesitamos conservarlos. Para indicar que queremos encontrar caracteres 
que no coincidan con el patrón, coloca un signo de intercalación ^ al comienzo de la secuencia. Así es como se verá:'''


pattern = r"[^a-zA-Z']"
text = re.sub(pattern, " ", text)
print(text)



'''
Ahora solo nos quedan letras, apóstrofos y espacios, aunque al parecer tenemos más espacios de los que necesitamos. 
En el siguiente paso, vamos a eliminar los espacios adicionales, ya que pueden entorpecer nuestro análisis. 
Podemos eliminarlos usando una combinación de los métodos join() y split().

Podemos usar el método split() para convertir nuestra cadena en una lista. Si llamamos a split() sin argumentos,
este divide el texto en los espacios o grupos de espacios:
'''

text = text.split()

print(text)

#El resultado es una lista sin espacios:

#RESULTADO :['I', 'liked', 'this', 'show', 'from', 'the', 'first', 'episode', 'I', 'saw', 'which', 'was', 'the', 'Rhapsody', 'in', 'Blue', 'episode', 'for', 'those', 'that', "don't", 'know', 'what', 'that', 'is', 'the', 'Zan', 'going', 'insane', 'and', 'becoming', 'pau', 'lvl', 'ep', 'Best', 'visuals', 'and', 'special', 'effects', "I've", 'seen', 'on', 'a', 'television', 'series', 'nothing', 'like', 'it', 'anywhere']

#luego recombinamos estos elementos en una cadena con espacios utilizando el método join():
text = " ".join(text)
print(text)


#Entonces obtenemos una línea sin espacios adicionales:

# RESULTADO "I liked this show from the first episode I saw which was the Rhapsody in Blue episode for those that don't know what that is the Zan going insane and becoming pau lvl ep Best visuals and special effects I've seen on a television series nothing like it anywhere"



#EJERCICIO 

'''
Escribe la función clear_text(text) para mantener solo letras latinas, espacios y apóstrofos en el texto. 
También elimina cualquier espacio adicional. La función tomará el texto inicial y devolverá el texto después de limpiarlo.

Imprime el texto inicial y el texto después de la limpieza y lematización (en precódigo).

'''

try: 
    import random  # para seleccionar una reseña aleatoria
    import pandas as pd

    import spacy
    import re

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    data = pd.read_csv('/datasets/imdb_reviews_small.tsv', sep='\t')
    corpus = data['review']


    def clear_text(text):

        clean_text = re.sub(r'[^a-zA-z\']', ' ', text)
        clean_text = " ".join(clean_text.split())

        return clean_text


    def lemmatize(text):

        doc = nlp(text.lower())

        lemmas = []
        for token in doc:
            lemmas.append(token.lemma_)
        return ' '.join(lemmas)


    # guarda el índice de revisión en la variable review_idx
    # ya sea como un número aleatorio o un valor fijo, por ejemplo, 2557
    review_idx = random.randint(0, len(corpus) - 1)
    # review_idx = 2557

    review = corpus[review_idx]

    print('El texto original:', review)
    print()
    print('El texto lematizado:', lemmatize(clear_text(review)))

except: print("No hay ruta aaa g ")


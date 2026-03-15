#TF-IDF

'''
Docstring for TF-IDF_teoria

La bolsa de palabras toma en cuenta la frecuencia de las palabras. 
Sin embargo, este sistema a veces puede crear un problema cuando una palabra en particular no se usa con frecuencia, 
pero aún contiene mucha información. En este caso, la bolsa de palabras puede fallar al priorizar una palabra importante. 
Vamos a averiguar cuántas veces aparece una palabra única en el corpus y en sus textos individuales.


La importancia de una palabra dada se determina por el valor de TF-IDF (frecuencia de término - frecuencia inversa de documento). 
TF es la frecuencia con la que aparece una palabra en un texto, mientras que IDF mide su frecuencia de aparición en el corpus.

Esta es la fórmula para TF-IDF:

                    TF-IDF = TF * IDF

Así es como se calcula TF:

                    TF = t / n

En la fórmula, t (término) es el número de ocurrencias de la palabra y n es el número total de palabras en el texto.

El papel de la IDF en la fórmula es reducir el peso de las palabras más utilizadas en cualquier otro texto del corpus dado. 
La IDF depende del número total de textos en un corpus (D) y del número de textos donde aparece la palabra (d).

                    IDF = log10(D / d)

Consideremos un corpus que consiste en veinte poemas. El primer poema tiene 40 palabras. 
Nos interesa la palabra "río", la cual aparece cinco veces en el poema. El corpus contiene dos poemas con la palabra "río". 
Vamos a averiguar el TF-IDF para "río" en el primer poema del corpus.

La TF es:

TF = t / n = 5 / 40 = 0.125

Dos de veinte poemas del corpus contienen "río". Entonces, la IDF es igual a:

IDF = log10(20 / 2) = 1

Y esto es lo que obtenemos para TF-IDF:

TF-IDF = TF * IDF = 0.125 * 1 = 0.125

Cuanto mayor sea el valor de TF-IDF, más única será la palabra en comparación con el resto del corpus. Cuanto menos frecuentemente aparece una palabra en los textos del corpus, mayor será su valor de TF-IDF.
'''

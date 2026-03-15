#BERT y preprocesamiento

'''

Vamos a resolver una tarea de clasificación para reseñas de películas usando la representación del lenguaje BERT, es decir, usando BERT para crear vectores para palabras. 
Vamos a tomar un modelo previamente entrenado llamado bert-base-uncased (entrenado en textos en inglés en minúsculas).

Para esta lección, usaremos un código prefabricado que no necesita cambios. Esta práctica no tiene nada de malo. 
Es común que los programadores copien y usen fragmentos de código existentes. Tu tarea será hacer que funcione.

Entonces, ¿cuál es la tarea? Tenemos un gran conjunto de datos de reseñas de películas y necesitamos entrenar la máquina para diferenciar entre reseñas positivas y negativas.

Vamos a resolver esta tarea usando las librerías PyTorch y transformers. 
La primera librería se utiliza para trabajar con modelos de redes neuronales, mientras que la segunda implementa BERT y otros modelos de representación del lenguaje. 
Vamos a importarlas:

'''

import numpy as np
import torch
import transformers

'''
Antes de convertir textos en vectores, necesitamos preprocesar el texto. BERT tiene su propio tokenizador basado en el corpus en el que fue entrenado. 
Otros tokenizadores no funcionan con BERT y no requieren lematización.

Pasos de preprocesamiento para el texto:

'''
#1 Inicializa el tokenizador como una instancia de BertTokenizer() con el nombre del modelo previamente entrenado.
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

#2 Convierte el texto en ID de tokens y el tokenizador BERT devolverá los ID de tokens en lugar de los tokens:

example = 'Es muy práctico utilizar transformadores'
ids = tokenizer.encode(example, add_special_tokens=True)
print(ids)

'[101, 2009, 2003, 2200, 18801, 2000, 2224, 19081, 102] <---- Resultados'


#3
'''
Considera que los identificadores de token anteriores son esencialmente índices numéricos para tokens en el diccionario interno utilizado por BERT. 
También debes saber que el diccionario se usó para entrenar BERT previamente, es parte del modelo BERT y está cargado con el método from_pretrained.

Para operar el modelo correctamente, establecemos el argumento add_special_tokens en True. Significa que agregamos el token inicial (101) y el token final (102) a cualquier texto que se esté transformando.
BERT acepta vectores de una longitud fija, por ejemplo, de 512 tokens. 
Si no hay suficientes palabras en una cadena de entrada para completar todo el vector con tokens (o, más bien, sus identificadores), 
el final del vector se rellena con ceros. Si hay demasiadas palabras y la longitud del vector excede 510 (recuerda que se reservan dos posiciones para los tokens de inicio y finalización), 
o bien la cadena de entrada se limita al tamaño de 510, o bien se suelen omitir algunos identificadores devueltos por tokenizer.encode(), 
por ejemplo, todos los identificadores después de la posición 512 en la lista:
'''
n = 512

padded = np.array(ids[:n] + [0]*(n - len(ids)))

print(padded)


#Obtenemos:

'[101 2009 2003 2200 18801 2000 2224 19081 102 0 0 0 0 ... 0 ]'

'''
Ahora tenemos que decirle al modelo por qué los ceros no tienen información significativa. 
Esto es necesario para el componente del modelo que se llama attention. 
Vamos a descartar estos tokens y crear una máscara para los tokens importantes, indicando valores cero y distintos de cero:
'''
attention_mask = np.where(padded != 0, 1, 0)
print(attention_mask.shape)


#Obtenemos:
'(512, )'

'''
Podemos establecer manualmente la longitud máxima de la entrada. 
Para ello, establece el parámetro max_length en el valor deseado y especifica truncation=True. 
Esto cortará los tokens que excedan el límite. Ten en cuenta que la parte resultante del texto inicial puede ser incluso más pequeña si add_special_token es True.
Considera este ejemplo intacto:
'''
example = 'It is very handy to use transformers'
ids = tokenizer.encode(example, add_special_tokens=True)
print(ids)


#resultado
'[101, 2009, 2003, 2200, 18801, 2000, 2224, 19081, 102]'

#y el truncado:
example = 'It is very handy to use transformers'
ids = tokenizer.encode(example, add_special_tokens=True, 
               max_length=5, truncation=True)
print(ids)
#resultado:
'[101, 2009, 2003, 2200, 102]'

#Umbral de clasificación
'''
¿Qué podemos hacer para optimizar el entrenamiento de la regresión logística? Deberíamos echar un vistazo a cómo funciona internamente.
Para determinar la respuesta, la regresión logística calcula la probabilidad de cada clase.
 Dado que solo tenemos dos clases (cero y uno), la probabilidad de la clase "1" es la que nos interesa. Esta probabilidad varía de cero a uno: si es mayor a 0.5, la observación se clasifica como positiva; si es menor, como negativa.

El punto de corte entre clasificaciones positivas y negativas se llama umbral. Por defecto es 0.5, pero ¿qué tal si lo cambiamos?
'''

'''
¿Cómo afecta la reducción del umbral a la precisión y a recall (sensibilidad)?
-La precisión disminuirá, recall aumentará
'''
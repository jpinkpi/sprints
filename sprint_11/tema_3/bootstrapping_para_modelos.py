#Bootstrapping para modelos

'''
Channel Tunnel es una escuela de francés que ofrece clases exprés de francés a un precio razonable. Está en el proceso de desarrollo de un modelo que determine la probabilidad de que un alumno asista o no a una clase. Diariamente se reciben muchas solicitudes. La prioridad se asigna por orden de solicitud, o sea, la primera en llegar, será la primera en ser atendida. Aproximadamente la mitad de quienes solicitan una clase, no se presentan y no la pagan. La dirección de la escuela ha decidido impartir clases solo a estudiantes con mayor probabilidad de asistir a clase. Debido a los posibles riesgos para su reputación, la empresa introducirá el nuevo sistema únicamente a condición de que se demuestre el aumento de los ingresos. Para tomar la decisión correcta, hay que evaluar la distribución de probabilidad de los ingresos.

Aquí tienes las condiciones importantes para la tarea:

El modelo de predicción de la probabilidad de asistencia a clase ya está entrenado. Las predicciones se encuentran en el archivo eng_probabilities.csv, y las respuestas correctas en eng_target.csv.
El coste de una lección es de 10 dólares. Se pueden programar hasta 10 lecciones por día. Los ingresos diarios actuales ascienden a 50 dólares (la mitad de los estudiantes cancelan la lección).
La media diaria de solicitudes recibidas es de 25.
El ingreso objetivo para la implementación del nuevo sistema se establece en 75 dólares, y la probabilidad de alcanzar este objetivo debe ser de al menos el 99%.

'''

#Ejercicio 1
'''
Escribe la función revenue() que calcula y devuelve el valor de los ingresos. Esta función utiliza:

la lista de respuestas (target): si el estudiante asistirá a la clase
la lista de probabilidades (probabilities): el modelo evalúa si el estudiante asistirá o no
el número de estudiantes que asisten a las clases por día (count).
La función debe seleccionar a los estudiantes con mayor probabilidad de asistir a clase y, basándose en las respuestas, calcular los posibles ingresos. 
Ten en cuenta que la función toma series de datos así que no es necesario utilizar ningún dataset para esta tarea.

En el precódigo tenemos un ejemplo de ejecución de la función en el que las listas de respuestas y probabilidades son cortas y el número de estudiantes es solo de 3.
'''

import pandas as pd

def revenue(target, probabilities, count):
    probs_sorted = probabilities.sort_values(ascending = False)
    selected = target[probs_sorted.index][:count]# < escribe tu código aquí >
    return 10 * selected.sum()# < escribe tu código aquí >

target = pd.Series([1,   1,   0,   0,  1,    0])
probab = pd.Series([0.2, 0.9, 0.8, 0.3, 0.5, 0.1])

res = revenue(target, probab, 3)

print(res)


#Ejercicio 2
'''
Para encontrar el cuantil de ingresos del 1 %, realiza el proceso de bootstrapping con 1000 repeticiones.

Guarda la lista de estimaciones del bootstrapping en la variable values y el cuantil del 1 % en la variable lower. 
Imprime los ingresos promedio y el cuantil del 1 % (en precódigo)

'''
try:
    import pandas as pd
    import numpy as np

    # Abre los archivos
    # toma el índice “0” para convertir los datos a pd-Series
    target = pd.read_csv('/datasets/eng_target.csv')['0']
    probabilities = pd.read_csv('/datasets/eng_probabilites.csv')['0']

    def revenue(target, probabilities, count):
        probs_sorted = probabilities.sort_values(ascending=False)
        selected = target[probs_sorted.index][:count]
        return 10 * selected.sum()

    state = np.random.RandomState(12345)
        
    values = []
    for i in range(1000):
        target_subsample = target.sample(n=25,replace=True, random_state=state)
        probs_subsample = probabilities[target_subsample.index]
        # < escribe tu código aquí >
        
        values.append(revenue(target_subsample, probs_subsample, 10))# < escribe tu código aquí >)

    values = pd.Series(values)
    lower = values.quantile(q=.01)# < escribe tu código aquí >

    mean = values.mean()
    print("Ingresos promedio:", mean)
    print("Cuantil del 1 %:", lower)
except: print("prueba")
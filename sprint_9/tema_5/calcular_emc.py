#Calcular el EMC
'''
Vamos a automatizar el cálculo del EMC.

¿Cómo calculaste el EMC en el ejercicio pasado?
¿Lo hiciste manualmente con una calculadora o programaste un sencillo programa en Python?
'''


#EJERCICIO 1
'''
Escribe una función mse(). 
Debe tomar respuestas correctas y predicciones y devolver el valor del error cuadrático medio.

Tomamos las respuestas (costos reales) y las predicciones de la tabla.
Imprime en la pantalla el valor del EMC.
'''

def mse(answers, predictions):
    sq_errors = []
    for i in range (len(answers)):
        error= (answers[i] - predictions[i])**2
        sq_errors.append(error)
    return sum(sq_errors)/len(answers)# < escribe tu código aquí  >
    
answers = [623, 253, 150, 237]
predictions = [649, 253, 370, 148]

print(mse(answers, predictions))

#EJERCICIO 2
'''
También hay una función para calcular el EMC en sklearn.

Busca en la documentación el nombre de la función y cómo opera. 
Impórtala y resuelve el mismo problema. Imprime en la pantalla el valor del EMC.

Aquí puedes encontrar la documentación de la librería sklearn: https://scikit-learn.org/stable/modules/classes.html. 
Si pierdes el vínculo, haz una búsqueda con "sklearn reference" como frase clave.
'''

from sklearn.metrics import mean_squared_error# < importa la función de cálculo del EMC desde la librería scikit-learn>

answers = [623, 253, 150, 237]
predictions = [649, 253, 370, 148]

result =mean_squared_error(answers, predictions) # < llama a la función de cálculo de EMC >
print(result)



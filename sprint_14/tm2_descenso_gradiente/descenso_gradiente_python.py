#Vamos a escribir un algoritmo de descenso de gradiente en Python.


'''
Pasos del algoritmo:
1. Establecer el valor inicial, x⁰.
2. Calcular el gradiente de la función de pérdida:
   ∇f(x) = (∂f/∂x₁ , ∂f/∂x₂ , … , ∂f/∂xₙ)
3. Actualizar el valor con la fórmula:
   x₁ = x₀ + μ x ( -∇f(x) )
   donde μ es el tamaño de paso.
4. Repetir durante el número de iteraciones especificado.

Función objetivo a minimizar:
   f(x₁, x₂) = (x₁ + x₂ - 1)² + (x₁ - x₂ - 2)²

El mínimo de esta función está en el punto (1.5 , -0.5).
'''

#Ejercicios 

#1
'''
Escribimos la función f. En el código, se denomina func(). Escribe la función gradient(), que calculará su gradiente según la fórmula. 
Prueba la función con varios vectores (en precódigo).
'''

import numpy as np


def func(x):
    return (x[0] + x[1] - 1) ** 2 + (x[0] - x[1] - 2) ** 2


def gradient(x):
    return np.array(np.array([4*x[0]-6, 4*x[1]+2]))# < escribe tu código aquí >)

        
print(gradient(np.array([0, 0])))
print(gradient(np.array([0.5, 0.3])))

#2
"""
Escribe la función gradient_descent() que aplica el algoritmo de descenso de gradiente a la función f(x). Esta función utiliza:

initialization — valor inicial del vector x
step_size — tamaño del paso μ
iterations — número de iteraciones
La función devuelve los valores del vector x tras haber realizado el número de iteraciones especificado. 
Prueba la función con un número diferente de iteraciones (en precódigo)
"""


import numpy as np


def func(x):
    return (x[0] + x[1] - 1) ** 2 + (x[0] - x[1] - 2) ** 2


def gradient(x):
    return np.array([4 * x[0] - 6, 4 * x[1] + 2])

def gradient_descent(initialization, step_size, iterations):
    x = initialization.astype("float")
    for _ in range(iterations):
        x = x - step_size * gradient(x)
    return x 
    # < escribe tu código aquí >


print(gradient_descent(np.array([0, 0]), 0.1, 5))
print(gradient_descent(np.array([0, 0]), 0.1, 100))
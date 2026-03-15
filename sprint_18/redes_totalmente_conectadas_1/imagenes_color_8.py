#Imagenes a color
'''

Ahora que hemos terminado con blanco y negro, seguiremos con las imágenes en color.
Las imágenes en color o imágenes RGB consisten en tres canales: rojo, verde y azul. 
De hecho, dichas imágenes son matrices tridimensionales 
con celdas que contienen números enteros en el rango de 0 a 255.

En NumPy, las matrices tridimensionales trabajan de la misma forma que las bidimensionales.

Compara cómo se crean:

np.array([[0, 255],
                    [255, 0]])
np.array([[[0, 255, 0], [128, 0, 255]], 
                    [[12, 89, 0], [5,  89, 245]]])
En una matriz tridimensional obtenida de una imagen, todo es casi igual. 
La primera coordenada es la ID de fila y la segunda es la ID de columna. 
Pero aquí también tenemos una nueva tercera coordenada que indica el canal.

Entonces, una matriz tridimensional es como una matriz bidimensional de una imagen en blanco 
y negro. La única diferencia es que cada pixel de dicha matriz almacena tres números que 
representan el brillo de cada uno de los tres canales: rojo, verde y azul.

¡Es hora de resolver algunas tareas! Esta foto de un gato genial se descargó de este sitio: 
www.petful.com (materiales en inglés).
'''
try:
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    image = Image.open(r"C:\Users\josep\Downloads\Image (3).png")
    array = np.array(image)

    plt.imshow(array)
except:print("prueba")

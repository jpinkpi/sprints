#EXACTITUD
'''
Nada es perfecto y todo es relativo. Incluso los mejores modelos cometen errores, pero el número de errores nos importa solo en relación con el número total de respuestas.
La relación entre el número de respuestas correctas y el número total de preguntas (es decir, el tamaño del conjunto de datos de prueba) se denomina exactitud (accuracy).
'''

#EJERCICI0
#Escribe la función accuracy(). Esta divide el número de respuestas correctas entre el número total de predicciones realizadas y devuelve la puntuación de exactitud.
#Muestra la exactitud en la pantalla de la siguiente manera (ya en el precódigo):

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(r"C:\Users\josep\Downloads\moved_train_data_us.csv")

df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

model = DecisionTreeClassifier(random_state=12345, class_weight='balanced')

model.fit(features, target)

test_df = pd.read_csv(r"C:\Users\josep\Downloads\moved_train_data_us.csv")

test_df.loc[test_df['last_price'] > 113000, 'price_class'] = 1
test_df.loc[test_df['last_price'] <= 113000, 'price_class'] = 0

test_features = test_df.drop(['last_price', 'price_class'], axis=1)
test_target = test_df['price_class']
test_predictions = model.predict(test_features)


def error_count(answers, predictions):
    count = 0
    for i in range(len(answers)):
        if answers[i] != predictions[i]:
            count += 1
    return count

def accuracy(answers, predictions):
    correct= 0
    for i in range(len(answers)):
         if answers[i] == predictions[i]:
             correct += 1
    return correct / len(answers)# < escribe el código aquí >

print('Accuracy:', accuracy(test_target, test_predictions))
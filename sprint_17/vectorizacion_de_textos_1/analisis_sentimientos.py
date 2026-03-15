
#Analisis de sentimiento
'''
Para determinar el tono del texto, vamos a usar valores TF-IDF como características.
El análisis de sentimiento identifica textos cargados de emociones. 
Esta herramienta puede ser extremadamente útil en los negocios al evaluar las reacciones de los consumidores ante un nuevo producto. Un humano necesitaría varias horas para analizar miles de reseñas, 
mientras que una computadora lo haría en un par de minutos.
El análisis de sentimiento funciona etiquetando el texto como positivo o negativo. 
Al texto positivo se le asigna un "1" y al texto negativo se le asigna un "0".
'''

#EJERCICIO
'''

Tu objetivo ahora es entrenar una regresión logística para determinar la tonalidad de las reseñas.

Tanto el dataset de entrenamiento como el conjunto de datos de prueba ya se han leído en el precódigo. Esto es lo que te pedimos que hagas:
Extrae reseñas lematizadas que se utilizarán para propósitos de entrenamiento y guárdalos en la variable train_corpus. Es importante observar que las reseñas lematizadas están en la columna review_lemm del dataset, así que no tendrás que lematizar reseñas por tu cuenta.
Establece las palabras vacías y guárdalas en la variable stop_words.
Inicializa el TF_IDF vectorizer y guárdalo en la variable count_tf_idf.
Ajusta y transforma el corpus de entrenamiento, y guarda los resultados en la variable tf_idf.
Las características que se usarán para el entrenamiento son los valores almacenados en la variable tf idf, así que establece features train en ella.
Los objetivos se encuentran en la columna pos del conjunto de datos (0 - reseña negativa, 1 - reseña positiva). Extrae los objetivos para fines de entrenamientos utilizando el nombre de la columna y guárdalos en la variable target_train.
Al igual que en el primer punto, extrae las reseñas lematizadas para probarlas y guárdalas en la variable test_corpus.
Obtén las características para probar transformando las reseñas lematizadas utilizando TF_IDF vectorizer, que utilizaste para el entrenamiento. Almacena los resultados de la transformación en la variable features_test.
Inicializa el modelo de regresión logística en la variable model y ajústalo con las características de entrenamiento y los objetivos.
Obtén predicciones para las características de prueba y almacénalos en la variable pred_test.
Las predicciones resultantes serán verificadas y si la precisión que alcanza tu modelo excede el 82%, se aceptará tu solución.
'''

try:
    import pandas as pd
    from nltk.corpus import stopwords as nltk_stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.linear_model import LogisticRegression

    train_data = pd.read_csv('/datasets/imdb_reviews_small_lemm_train.tsv', sep='\t')
    test_data = pd.read_csv('/datasets/imdb_reviews_small_lemm_test.tsv', sep='\t')

    train_corpus = train_data['review_lemm']
    stop_words = set(nltk_stopwords.words('english'))

    count_tf_idf = TfidfVectorizer(stop_words=stop_words)
    tf_idf = count_tf_idf.fit_transform(train_corpus)

    features_train = tf_idf
    target_train = train_data['pos']

    test_corpus = test_data['review_lemm']
    features_test = count_tf_idf.transform(test_corpus)

    model = LogisticRegression()
    model.fit(features_train, target_train)
    pred_test = model.predict(features_test)

    submission = pd.DataFrame({'pos':pred_test})
    print(submission)
except: print("Prueba")


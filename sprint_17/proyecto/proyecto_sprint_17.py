#PROYECTO SPRINT 17
#////////////////////////////////////////////////////

#1 Iniciamos los datos
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tqdm.auto import tqdm
import sklearn.metrics as metrics 
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (10, 6)

# esto es para usar progress_apply, puedes leer más en https://pypi.org/project/tqdm/#pandas-integration
tqdm.pandas() #Simplemente visualiza el porcentaje de un proceso realizado en un barra.

#EJEMPLO 
"df['longitud'] = df['review'].progress_apply(len)"
#////////////////////////////////////////////////////////////////////////


#2 Cargamos los datos
df_reviews= pd.read_csv(r"C:\Users\josep\Downloads\imdb_reviews.tsv", sep="\t",dtype={'votes': 'Int64'})
#print(df.sample(n=4))
df_reviews.info()

#//////////////////////////////////////////////////////////////////////////

#3 EDA
"Veamos el número de peliculas y reseñas a lo largo de los años "

fig, axs = plt.subplots(2, 1, figsize=(16, 8))

ax = axs[0]

dft1 = df_reviews[['tconst', 'start_year']].drop_duplicates() \
    ['start_year'].value_counts().sort_index()
dft1 = dft1.reindex(index=np.arange(dft1.index.min(), max(dft1.index.max(), 2021))).fillna(0)
dft1.plot(kind='bar', ax=ax)
ax.set_title('Número de películas a lo largo de los años')

ax = axs[1]

dft2 = df_reviews.groupby(['start_year', 'pos'])['pos'].count().unstack()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)

dft2.plot(kind='bar', stacked=True, label='#reviews (neg, pos)', ax=ax)

dft2 = df_reviews['start_year'].value_counts().sort_index()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)
dft3 = (dft2/dft1).fillna(0)
axt = ax.twinx()
dft3.reset_index(drop=True).rolling(5).mean().plot(color='orange', label='reviews per movie (avg over 5 years)', ax=axt)

lines, labels = axt.get_legend_handles_labels()
ax.legend(lines, labels, loc='upper left')

ax.set_title('Número de reseñas a lo largo de los años')

fig.tight_layout()
plt.show()

#Veamos la distribución del número de reseñas por película con el conteo exacto y KDE (solo para saber cómo puede diferir del conteo exacto)


fig, axs = plt.subplots(1, 2, figsize=(16, 5))

ax = axs[0]
dft = df_reviews.groupby('tconst')['review'].count() \
    .value_counts() \
    .sort_index()
dft.plot.bar(ax=ax)
ax.set_title('Gráfico de barras de #Reseñas por película')

ax = axs[1]
dft = df_reviews.groupby('tconst')['review'].count()
sns.kdeplot(dft, ax=ax)
ax.set_title('Gráfico KDE de #Reseñas por película')

fig.tight_layout()
plt.show()

print(df_reviews['pos'].value_counts())



fig, axs = plt.subplots(1, 2, figsize=(12, 4))

ax = axs[0]
dft = df_reviews.query('ds_part == "train"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('El conjunto de entrenamiento: distribución de puntuaciones')

ax = axs[1]
dft = df_reviews.query('ds_part == "test"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('El conjunto de prueba: distribución de puntuaciones')


fig.tight_layout()
plt.show()

"Distribución de reseñas negativas y positivas a lo largo de los años para dos partes del conjunto de datos"

fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw=dict(width_ratios=(2, 1), height_ratios=(1, 1)))

ax = axs[0][0]

dft = df_reviews.query('ds_part == "train"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('El conjunto de entrenamiento: número de reseñas de diferentes polaridades por año')

ax = axs[0][1]

dft = df_reviews.query('ds_part == "train"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('El conjunto de entrenamiento: distribución de diferentes polaridades por película')

ax = axs[1][0]

dft = df_reviews.query('ds_part == "test"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('El conjunto de prueba: número de reseñas de diferentes polaridades por año')

ax = axs[1][1]

dft = df_reviews.query('ds_part == "test"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('El conjunto de prueba: distribución de diferentes polaridades por película')

fig.tight_layout()
plt.show()

#4 Procedimiento de evalucación 
"Composición de una rutina de evaluación que se pueda usar para todos los modelos en este proyecto"

import sklearn.metrics as metrics
def evaluate_model(model, train_features, train_target, test_features, test_target):

    eval_stats = {}

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):

        eval_stats[type] = {}

        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]

        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]

        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps

        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # Valor F1
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'Valor F1')

        # ROC
        ax = axs[1]
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')
        ax.set_title(f'Curva ROC')

        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)

    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'F1', 'APS', 'ROC AUC'))

    print(df_eval_stats)

    return
#////////////////////////////////////////////////////////////////////////////////////////
# 5 Normalización
"Suponemos que todos los modelos a continuación aceptan textos en minúsculas y sin dígitos, signos de puntuación, etc."
def normalizar_texto(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text) #eliminamos los digitos
    text = re.sub(r'[^\w\s]', ' ', text) #eliminamos las puntuaciones
    text = re.sub(r'\s+', ' ', text).strip() #Eliminamos espacios extra
    return text

df_reviews ["review_norm"] = df_reviews["review"].progress_apply(normalizar_texto)

"""
Se normalizó el texto convirtiéndolo a minúsculas y eliminando dígitos,
signos de puntuación y espacios redundantes. No se aplicó lematización
ni stemming para mantener la semántica original del texto y reducir
complejidad computacional.
"""

#///////////////////////////////////////////////////////////////////////////////////
#6 División entrenamiento / prueba
"Por fortuna, todo el conjunto de datos ya está dividido en partes de entrenamiento/prueba; 'ds_part' es el indicador correspondiente."
df_reviews_train = df_reviews.query('ds_part == "train"').copy()
df_reviews_test = df_reviews.query('ds_part == "test"').copy()

train_features = df_reviews_train["review_norm"]
test_features = df_reviews_test["review_norm"]

train_target = df_reviews_train['pos']
test_target = df_reviews_test['pos']



print(df_reviews_train.shape)
print(df_reviews_test.shape)
"""
Se utiliza la división entrenamiento/prueba proporcionada en el dataset
para evitar fuga de información y mantener consistencia experimental.
"""
#//////////////////////////////////////////////////////////////////////////////////
#7 Trabajar con Modelos 

#7.1 Modelo 0 constante
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy= "most_frequent")
dummy.fit(df_reviews_train, train_target)

evaluate_model(dummy, train_features, train_target, test_features, test_target)


"""
El modelo constante se utilizó como línea base. Como se esperaba, su
desempeño es equivalente al azar (ROC AUC ≈ 0.5) y el valor F1 es nulo,
ya que predice siempre la clase mayoritaria. Esto confirma que los
modelos posteriores deberán aprender patrones reales del texto para
superar esta referencia.
"""

# Modelo 1 - NLTK, TF-IDF y LR

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import stopwords

stop_words = set(nltk_stopwords.words('english'))
stop_words = stopwords.words('english')
tfidf = TfidfVectorizer(stop_words=stop_words, ngram_range=(1,2), min_df=5, max_df=.9)      #Min_df = elimina ruido      max_df = elimina palabras demasiado comunes    


X_train_tfidf = tfidf.fit_transform(train_features)
X_test_tfidf = tfidf.transform(test_features)

# Regresion Lineal

lr = LogisticRegression(max_iter=100, solver='liblinear', random_state=42)
lr.fit(X_train_tfidf, train_target)
#evaluate_model(lr, 
#               X_train_tfidf, 
 #              train_target,
  #               X_test_tfidf, 
  #               test_target)

#Modelo 3 spaCy, TF-IDF y LR

import spacy


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
print("Modelo cargado correctamente")
def text_preprocessing_3(text):

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    #tokens = [token.lemma_ for token in doc]

    return ' '.join(tokens)


df_reviews_train["review_spacy"] = df_reviews_train["review_norm"].progress_apply(text_preprocessing_3)
df_reviews_test["review_spacy"] = df_reviews_test["review_norm"].progress_apply(text_preprocessing_3)

# TF-IDF
tfidf_spacy = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9
)

X_train_spacy = tfidf_spacy.fit_transform(df_reviews_train["review_spacy"])
X_test_spacy = tfidf_spacy.transform(df_reviews_test["review_spacy"])

# Logistic Regression
lr_spacy = LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

lr_spacy.fit(X_train_spacy, train_target)

# Evaluación
#evaluate_model(
#    lr_spacy,
#    X_train_spacy, train_target,
 #   X_test_spacy, test_target
#)


'''
Se entrenaron tres modelos de clasificación de texto, incluyendo un modelo base constante y dos modelos basados en TF-IDF con regresión logística. 
El mejor desempeño se obtuvo con el modelo TF-IDF + Logistic Regression, alcanzando un valor F1 de 0.88 en el conjunto de prueba, superando el umbral requerido. 
El uso de lematización con spaCy no produjo una mejora significativa, lo que sugiere que la representación TF-IDF ya capturaba adecuadamente la información relevante del texto.
'''



#///////////////////////////////////////////////////////////////////////////////////////////////////////
# 8 Mis reseñas 

# puedes eliminar por completo estas reseñas y probar tus modelos en tus propias reseñas; las que se muestran a continuación son solo ejemplos

my_reviews = pd.DataFrame([
    'I did not simply like it, not my kind of movie.',
    'Well, I was bored and felt asleep in the middle of the movie.',
    'I was really fascinated with the movie',
    'Even the actors looked really old and disinterested, and they got paid to be in the movie. What a soulless cash grab.',
    'I didn\'t expect the reboot to be so good! Writers really cared about the source material',
    'The movie had its upsides and downsides, but I feel like overall it\'s a decent flick. I could see myself going to see it again.',
    'What a rotten attempt at a comedy. Not a single joke lands, everyone acts annoying and loud, even kids won\'t like this!',
    'Launching on Netflix was a brave move & I really appreciate being able to binge on episode after episode, of this exciting intelligent new drama.'
], columns=['review'])

"""
my_reviews = pd.DataFrame([
    'Simplemente no me gustó, no es mi tipo de película.',
    'Bueno, estaba aburrido y me quedé dormido a media película.',
    'Estaba realmente fascinada con la película',
    'Hasta los actores parecían muy viejos y desinteresados, y les pagaron por estar en la película. Qué robo tan desalmado.',
    '¡No esperaba que el relanzamiento fuera tan bueno! Los escritores realmente se preocuparon por el material original',
    'La película tuvo sus altibajos, pero siento que, en general, es una película decente. Sí la volvería a ver',
    'Qué pésimo intento de comedia. Ni una sola broma tiene sentido, todos actúan de forma irritante y ruidosa, ¡ni siquiera a los niños les gustará esto!',
    'Fue muy valiente el lanzamiento en Netflix y realmente aprecio poder seguir viendo episodio tras episodio de este nuevo drama tan emocionante e inteligente.'
], columns=['review'])
"""

my_reviews['review_norm'] = my_reviews['review'].apply(normalizar_texto)

my_reviews['review_spacy'] = my_reviews['review_norm'].apply(text_preprocessing_3)

#Vectorizacion
X_my_reviews = tfidf_spacy.transform(my_reviews['review_spacy'])

#prediccion

my_reviews['predicted_class'] = lr_spacy.predict(X_my_reviews)
my_reviews['predicted_proba'] = lr_spacy.predict_proba(X_my_reviews)[:, 1]


my_reviews[['review', 'predicted_class', 'predicted_proba']]
print(my_reviews)



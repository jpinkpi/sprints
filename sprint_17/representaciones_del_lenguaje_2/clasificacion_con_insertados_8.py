#Clasifiacion de Insertados
'''
Entrena un modelo de regresión logística mediante insertados. Imprime el valor de exactitud para el conjunto de entrenamiento.
Calcular los insertados de BERT lleva mucho tiempo (si no estás usando una GPU), por lo que te sugerimos tomar solo 200 elementos aleatorios del conjunto. 
Los elementos se guardan en el archivo imdb_reviews_200.tsv. Divídelos en conjuntos de entrenamiento y prueba en una proporción de 50:50 para una prueba correcta.

El objetivo se almacena en la variable pos.

'''
try:
    #Inicialización
    import numpy as np
    import pandas as pd

    import torch
    import transformers

    from tqdm.auto import tqdm

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split

    #Variables globales
    max_sample_size = 200

    # Cargar datos
    df_reviews = pd.read_csv('/datasets/imdb_reviews_200.tsv', sep='	')


    # Preprocesamiento para BERT
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    ids_list = []
    attention_mask_list = []

    max_length = 512

    for input_text in df_reviews.iloc[:max_sample_size]['review']:
        ids = tokenizer.encode(input_text.lower(), add_special_tokens=True, truncation=True, max_length=max_length)
        padded = np.array(ids + [0]*(max_length - len(ids)))
        attention_mask = np.where(padded != 0, 1, 0)
        ids_list.append(padded)
        attention_mask_list.append(attention_mask)


    # Obtener insertados

    config = transformers.BertConfig.from_pretrained('bert-base-uncased')
    model = transformers.BertModel.from_pretrained('bert-base-uncased')


    batch_size = 25    # por lo general, el tamaño del lote es igual a 100, pero lo podemos configurar en valores más bajos para reducir los requisitos de memoria

    embeddings = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Uso del dispositivo {device}.')
    model.to(device)

    for i in tqdm(range(len(ids_list) // batch_size)):
        
        ids_batch = torch.LongTensor(ids_list[batch_size*i:batch_size*(i+1)]).to(device)
        attention_mask_batch = torch.LongTensor(attention_mask_list[batch_size*i:batch_size*(i+1)]).to(device)

        with torch.no_grad():
            model.eval()
            batch_embeddings = model(ids_batch, attention_mask=attention_mask_batch)

        embeddings.append(batch_embeddings[0][:,0,:].detach().cpu().numpy())

    # Modelo
    features = np.concatenate(embeddings)
    target = df_reviews.iloc[:max_sample_size]['pos']

    print(features.shape)
    print(target.shape)

except:print("Prueba")


#Insertados Bert
''' 
Insertados de BERT
Supongamos que la lista de los ID de vector (relleno) y la lista de máscaras de atención se formaron de la siguiente manera:

ids_list = []
attention_mask_list = []

max_length = 512

for input_text in corpus[:200]:
    ids = tokenizer.encode(
        input_text.lower(),
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    padded = np.array(ids + [0] * (max_length - len(ids)))
    attention_mask = np.where(padded != 0, 1, 0)
    ids_list.append(padded)
    attention_mask_list.append(attention_mask)
Ya casi tenemos todo listo para formar vectores con el modelo BERT y clasificar las reseñas. Es hora de pasar a los tensores.

Inicializa la configuración BertConfig. Pásale un archivo JSON con la descripción de la configuración del modelo. JSON (notación de objetos de JavaScript) es un flujo de números, letras, dos puntos y corchetes que devuelve un servidor cuando se le llama.

Inicializa el modelo de la clase BertModel. Pasa el archivo con el modelo previamente entrenado y la configuración:

config = BertConfig.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
Vamos a comenzar por convertir textos en insertados. Esto puede tardar varios minutos, así que accede a la biblioteca tqdm (árabe: taqadum, تقدّم, “progreso”). Esta muestra el progreso de la operación. Luego simplemente envuelve tu bucle en tqdm(). Utiliza tqdm.auto para importar la opción correcta para tu plataforma. Mira cómo funciona:

from tqdm.auto import tqdm

for i in tqdm(range(int(8e6))):
    pass

# aparecerá la barra de progreso
El modelo BERT crea insertados en lotes. Haz pequeño el tamaño del lote para que la RAM no se vea abrumada:

batch_size = 100
Haz un bucle para los lotes. La función tqdm() indicará el progreso:

# creación de una lista vacía de insertados de reseñas
embeddings = []

for i in tqdm(range(len(ids_list) // batch_size)):
    ...
Transforma los datos en un formato de tensor. Tensor es un vector multidimensional en la librería Torch. El tipo de datos LongTensor almacena números en "formato largo", es decir, asigna 64 bits para cada número.

# unión de vectores de ids (de tokens) a un tensor
ids_batch = torch.LongTensor(ids_list[batch_size*i:batch_size*(i+1)])
# unión de vectores de máscaras de atención a un tensor
attention_mask_batch = torch.LongTensor(attention_mask_list[batch_size*i:batch_size*(i+1)])
Pasa los datos y la máscara al modelo para obtener insertados para el lote:

batch_embeddings = model(ids_batch, attention_mask=attention_mask_batch)
Utiliza la función  no_grad()(sin gradiente) para indicar que no necesitas gradientes en la librería Torch (al crear tu propio modelo BERT necesitas los gradientes para el modo de entrenamiento). Esta hará los cálculos más rápido:

with torch.no_grad():
    batch_embeddings = model(ids_batch, attention_mask=attention_mask_batch)
Extrae los elementos requeridos del tensor y agrega la lista de todos los insertados:

# convierte elementos de tensor a numpy.array con la función numpy()
embeddings.append(batch_embeddings[0][:,0,:].numpy())
Uniendo todo lo anterior, obtenemos este bucle:

batch_size = 100

embeddings = []

for i in tqdm(range(len(ids_list) // batch_size)):

    ids_batch = torch.LongTensor(
        ids_list[batch_size * i : batch_size * (i + 1)]
    )
    attention_mask_batch = torch.LongTensor(
        attention_mask_list[batch_size * i : batch_size * (i + 1)]
    )

    with torch.no_grad():
        batch_embeddings = model(
            ids_batch, attention_mask=attention_mask_batch
        )
    embeddings.append(batch_embeddings[0][:, 0, :].numpy())
Llama a la función concatenate() para concatenar todos los insertados en una matriz de características:

features = np.concatenate(embeddings)

'''
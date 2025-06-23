import os
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.layers import TextVectorization
#https://huggingface.co/datasets/Nicky0007/titulos_noticias_rcn_clasificadas/tree/main

# 1. Cargar el dataset.
file_path = os.path.join(os.path.dirname(__file__), "noticias-train.csv")
df = pd.read_csv(file_path, header=None, usecols=[0, 1], names=['message', 'label'], encoding='utf-8')

# 2. Pasar categorías de texto a números y a numpy.
categorias = sorted(df['label'].unique())
categoria_to_index = {nombre: idx for idx, nombre in enumerate(categorias)}
index_to_categoria = {idx: nombre for nombre, idx in categoria_to_index.items()}
df['label'] = df['label'].map(categoria_to_index)
X = df['message'].to_numpy()
y = df['label'].to_numpy()

# 3. Mezclar índices.
rng = np.random.default_rng(seed=43)
indices = rng.permutation(len(X))
X = X[indices]
y = y[indices]

# 4. 80/20 en entrenamiento y prueba.
index_split = int(len(X) * 0.8)
X_train, X_test = X[:index_split], X[index_split:]
y_train, y_test = y[:index_split], y[index_split:]

# 5. Vectorizar los mensajes.
vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=100)
vectorizer.adapt(X_train)

X_train_vec = vectorizer(X_train)
X_test_vec = vectorizer(X_test)
#print(f"Forma de X_train_vec: {X_train_vec[0], X_train[0]}")

# 6. Estructura de la red neuronal usando softmax.
num_clases = len(categorias)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=20000, output_dim=64),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_clases, activation='softmax')
])

# 7. Compilar la red neuronal.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 8. Entrenar el modelo.
model.fit(X_train_vec, y_train, epochs=10, validation_split=0.1)

# 9. Evaluar el modelo.
loss, acc = model.evaluate(X_test_vec, y_test)
print(f"Precision en test: {acc * 100:.2f}%")
print(f"Perdida en test: {loss * 100:.2f}%")

# 10. Interacción.
while True:
    message = input("Mensaje: ")
    if message.lower() == 'bye': break
    message_vec = vectorizer([message])
    prediction = model.predict(message_vec)[0]
    clase_predicha = np.argmax(prediction)
    confianza = prediction[clase_predicha]
    print(f"--------------------------- Categoría: {index_to_categoria[clase_predicha]} ({confianza*100:.2f}%) ---------------------------")
    print(f"Probabilidades: {', '.join([f'{index_to_categoria[i]}: {prediction[i]*100:.2f}%' for i in range(num_clases)])}")
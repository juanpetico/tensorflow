import os
import pandas as pd
import tensorflow as tf
from keras.layers import TextVectorization
import numpy as np
#https://www.kaggle.com/datasets/luistestalter2025/spam-or-ham-in-spanish


# 1. Cargar el dataset.
file_path = os.path.join(os.path.dirname(__file__), "spam.csv")
df = pd.read_csv(file_path, header=None, usecols=[0, 1], names=['message', 'label'], encoding='latin1')
X = df['message'].to_numpy()
y = df['label'].to_numpy()

# 2. Mezclar índices.
rng = np.random.default_rng(seed=42) 
indices = rng.permutation(len(X))
X = X[indices]
y = y[indices]

# 3. 80/20 en entrenamiento y prueba.
index_split = int(len(X) * 0.8)
X_train, X_test = X[:index_split], X[index_split:]
y_train, y_test = y[:index_split], y[index_split:]

# 4. 80/20 en entrenamiento y prueba.
vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=100)
vectorizer.adapt(X_train) # Crea un vocabulario con las palabras más frecuentes

X_train_vec = vectorizer(X_train)
X_test_vec = vectorizer(X_test)
#print(f"Forma de X_train_vec: {X_train_vec[0], X_train[0]}")

# 5. Estructura de la red neuronal usando sigmoide.
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 6. Compilar la red neuronal.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Entrenar el modelo.
history = model.fit(X_train_vec, y_train, epochs=10, validation_split=0.2)

# 8. Evaluar el modelo.
loss, acc = model.evaluate(X_test_vec, y_test)
print(f"Precision en test: {acc * 100:.2f}")
print(f"Perdida en test: {loss * 100:.2f}")

# 9. Interacción.
while True: 
    message = input("Mensaje: ")
    if message.lower() == 'bye': break
    message_vec = vectorizer([message])
    prediction = model.predict(message_vec)
    if prediction[0][0] > 0.5:
        print(f"El mensaje es SPAM. Tiene una prediccion del {prediction[0][0]*100:.2f}%")
    else:
        print(f"El mensaje es HAM. Tiene una prediccion del {prediction[0][0]*100:.2f}%")

# Hola Felipe, esta semana vamos a la universidad que tenemos trabajo -> HAM
# Has ganado!! Confirma y obten un 50%!  -> SPAM
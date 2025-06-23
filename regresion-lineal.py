import os
import pandas as pd
import numpy as np
import tensorflow as tf

# 1. Cargar el dataset.
file_path = os.path.join(os.path.dirname(__file__), "precios-casas.csv")
df = pd.read_csv(file_path)
df = df[['Dorms', 'Baths', 'Built Area', 'Total Area', 'Parking', 'Price_CLP']].dropna()

# 2. Entradas y salidas
X = df.drop('Price_CLP', axis=1).to_numpy().astype(np.float32)
y = df['Price_CLP'].to_numpy().astype(np.float32)

# 3. Normalizar datos.
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min)

# 4. 80/20 en entrenamiento y prueba.
split_idx = int(len(X_norm) * 0.8)
X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 5. Estructura de la red neuronal lineal.
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(1)
# ])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 6. Compilar la red neuronal.
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 7. Entrenar el modelo.
model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# 9. Evaluar el modelo.
loss, mae = model.evaluate(X_test, y_test)

# 10. Interacción.
while True:
    dorms = float(input("\nNº de dormitorios: "))
    baths = float(input("Nº de baños: "))
    built_area = float(input("Metros construidos: "))
    total_area = float(input("Metros totales de terreno: "))
    parking = float(input("Nº de estacionamientos: "))

    nuevo = np.array([[dorms, baths, built_area, total_area, parking]], dtype=np.float32)

    # Normalizar igual que el entrenamiento
    nuevo_norm = (nuevo - X_min) / (X_max - X_min)

    pred = model.predict(nuevo_norm)[0][0]
    print(f"\n Precio estimado: ${pred:,.0f} CLP")


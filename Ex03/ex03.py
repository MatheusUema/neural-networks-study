import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from PIL import Image

# Load MNIST
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

# Normalization
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

tf.random.set_seed(42)
model = tf.keras.Sequential()

# First layer
model.add(tf.keras.layers.Conv2D(32, (1, 1), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# Second layer
model.add(tf.keras.layers.Conv2D(64, (2, 2), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# Third layer
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# Flatten 
model.add(tf.keras.layers.Flatten())

# Densely connected layer
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Output layer - softmax
model.add(tf.keras.layers.Dense(10, activation='softmax'))

tf.keras.backend.clear_session()
tf.random.set_seed(42)

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# Model fit
history = model.fit(X_train, y_train, epochs=30, 
                    validation_data=(X_valid, y_valid))


pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.legend(loc="lower left")  # extra code
plt.savefig("keras_learning_curves_plot")  # extra code
plt.show()

model.evaluate(X_test, y_test)

# Selecting 4 images
sample_images = X_test[:4]
sample_labels = y_test[:4]


# Image saving
for i, img in enumerate(sample_images):
    img = np.squeeze(img)  
    img_path = f"digit_{i}.png"
    cv2.imwrite(img_path, img * 255) 
    print(f"Imagem {i} salva como {img_path}")

# Loading and proccessing inference
for i in range(4):
    img_path = f"digit_{i}.png"
    img = Image.open(img_path).convert('L') 
    img = np.array(img).reshape((1, 28, 28, 1)).astype('float32') / 255 
    
    # Realizar a predição
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    print(f"Imagem {i}: Predição do modelo = {predicted_label}, Rótulo verdadeiro = {sample_labels[i]}")

from tensorflow.keras import datasets, layers, models
import numpy as np
import cv2
from PIL import Image

# Carregar dataset MNIST
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Redimensionar e normalizar as imagens (0 a 1)
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Construir o modelo CNN
model = models.Sequential()

# Primeira camada de convolução e pooling
model.add(layers.Conv2D(32, (1, 1), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Segunda camada de convolução e pooling
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Terceira camada de convolução e pooling
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten para passar para camadas densamente conectadas
model.add(layers.Flatten())

# Camada densamente conectada
model.add(layers.Dense(128, activation='relu'))

# Camada de saída com softmax
model.add(layers.Dense(10, activation='softmax'))

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinando o modelo
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Selecionar 4 imagens do conjunto de teste para salvar
sample_images = test_images[:4]
sample_labels = test_labels[:4]

# Salvar as imagens
for i, img in enumerate(sample_images):
    img = np.squeeze(img)  # Remover a dimensão extra
    img_path = f"Ex03\digit_{i}.png"
    cv2.imwrite(img_path, img * 255)  # Reverter a normalização (0-255)
    print(f"Imagem {i} salva como {img_path}")

# Carregar e processar as imagens salvas para inferência
for i in range(4):
    img_path = f"Ex03\digit_{i}.png"
    img = Image.open(img_path).convert('L')

print(sample_labels)

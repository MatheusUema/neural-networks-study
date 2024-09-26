import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Dataset
df = pd.read_csv('Projeto 01 - MLP\default_features_1059_tracks.txt', delimiter=',', header=None)
X = df.iloc[:, :-2].values  # Features
y = df.iloc[:, -2:].values  # Lat&long

# Input scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Data split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Regression model
def create_regression_model(layers, neurons_per_layer, input_dim, output_dim, learning_rate=0.01, momentum=0.9):
    model = Sequential()
    
    model.add(Dense(neurons_per_layer[0], input_dim=input_dim, activation='relu'))

    if len(layers) > 1:
        model.add(Dense(neurons_per_layer[1], activation='relu'))

    model.add(Dense(output_dim))

    sgd = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae', 'mse'])
    
    return model

neurons_per_layer_reg = [64, 32]  # Middle layer neurons
output_dim_reg = 2  # Lat and long

# Parameters
layers = [1, 2]  # One or two middle layers
neurons_per_layer = [16, 8]  # Number of neurons per layer
learning_rate = 0.01
momentum = 0.9
epochs = 200
batch_size = 32

regression_model = create_regression_model([1, 2], neurons_per_layer_reg, input_dim=X_train.shape[1], output_dim=output_dim_reg, learning_rate=learning_rate, momentum=momentum)
history_regression = regression_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

test_loss_regression, test_mae, test_mse = regression_model.evaluate(X_test, y_test)
print(f"Erro Quadrático Médio (MSE) no teste: {test_mse:.4f}")

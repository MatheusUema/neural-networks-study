import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Wine dataset
data = load_wine()
X = data['data']
y = data['target']

# Normalizing input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Transforming output into categorical data
y_categorical = np.eye(len(np.unique(y)))[y]

# Split dataset into test and train
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=test_size, random_state=42)

# Neural network implementation
def create_model(layers, neurons_per_layer, input_dim, output_dim, learning_rate=0.01, momentum=0.9):
    model = Sequential()
    
    # Input layer and first middle layer
    model.add(Dense(neurons_per_layer[0], input_dim=input_dim, activation='relu'))

    # Optional second middle layer
    if len(layers) > 1:
        model.add(Dense(neurons_per_layer[1], activation='relu'))

    # Output layer with softmax for classification
    model.add(Dense(output_dim, activation='softmax'))

    # Model compiling SGD (gradient descent) e momentum
    sgd = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Model training
# Parameters
layers = [1, 2]  # One or two middle layers
neurons_per_layer = [16, 8]  # Number of neurons per layer
learning_rate = 0.01
momentum = 0.9
epochs = 200
batch_size = 32

# Model creation
model = create_model(layers, neurons_per_layer, input_dim=X_train.shape[1], output_dim=y_train.shape[1], learning_rate=learning_rate, momentum=momentum)

#  Model training
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Model evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Accuracy on test size: {test_accuracy*100:.2f}%")


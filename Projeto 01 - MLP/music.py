import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, layers, X, y, epochs, learning_rate, momentum, verbose=False):
        self.layers = layers
        self.weights, self.biases = self.initialize_weights()
        self.X = X
        self.y = y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.verbose=verbose

    def initialize_weights(self):
        weights = []
        biases = []
        for i in range(1, len(self.layers)):
            # Randomly initialized weights between layers i to i-1
            weights.append(np.random.randn(self.layers[i], self.layers[i - 1]) * 0.01) 
            biases.append(np.zeros((self.layers[i], 1)))
        return weights, biases

    # ReLU activation function
    def relu(self, l):
        return np.maximum(0, l)

    def relu_derivative(self, l):
        return (l > 0).astype(float)
    
    # Linear Activation for last layer
    def linear(self, l):
        return l

    def linear_derivative(self, l):
        return np.ones_like(l)

    def forward_propagation(self, X):
        activations = [X.T]  # First activation is input
        L_values = []  # Layer values L = W * A + b 

        # Forward for middle layers
        for i in range(len(self.weights) - 1):
            L = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            L_values.append(L)

            A = self.relu(L)

            activations.append(A)

        #Output layer
        L_output = np.dot(self.weights[-1], activations[-1]) + self.biases[-1]
        L_values.append(L_output)
        A_output = self.linear(L_output)

        activations.append(A_output)
        return L_values, activations
    
    def compute_loss(self, y_true, y_pred):
        loss = np.sqrt(np.mean(np.square(y_pred - y_true)))
        return loss

    def backward_propagation(self, L_values, activations):
        m = self.X.shape[0]
        dW, db = [], []
        dA_prev = activations[-1] - self.y

        # Momentum initial term
        momentum_dW = [np.zeros_like(w) for w in self.weights]
        momentum_db = [np.zeros_like(b) for b in self.biases]

        # Backpropagation for output layer
        dL = dA_prev
        dW_curr = np.dot(dL, activations[-2].T) / m
        db_curr = np.sum(dL, axis=1, keepdims=True) / m

        momentum_dW[-1] = self.momentum * momentum_dW[-1] + (1 - self.momentum) * dW_curr
        momentum_db[-1] = self.momentum * momentum_db[-1] + (1 - self.momentum) * db_curr

        dW.append(momentum_dW[-1])
        db.append(momentum_db[-1])

        # Backprop for middle layers
        for i in reversed(range(len(self.weights) - 1)):
            dA = np.dot(self.weights[i + 1].T, dL)
            dL = dA * self.relu_derivative(L_values[i])
            dW_curr = np.dot(dL, activations[i].T) / m
            db_curr = np.sum(dL, axis=1, keepdims=True) / m

            momentum_dW[i] = self.momentum * momentum_dW[i] + (1 - self.momentum) * dW_curr
            momentum_db[i] = self.momentum * momentum_db[i] + (1 - self.momentum) * db_curr

            dW.append(momentum_dW[i])
            db.append(momentum_db[i])

        dW.reverse()
        db.reverse()

        # Weights update
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def train(self):
        losses = []
        for epoch in range(self.epochs):
            # Forward propagation
            L_values, activations = self.forward_propagation(self.X)

            # Loss 
            loss = self.compute_loss(self.y, activations[-1])
            losses.append(loss)

            # Backpropagation
            self.backward_propagation(L_values, activations)
            
            if epoch % 100 == 0 and self.verbose:
                print(f"Epoch {epoch}, MSE Loss: {loss:.4f}")
        return losses

    def predict(self, X):
        _, activations = self.forward_propagation(X)
        return activations[-1]

def plot_results(layers, layers_2, X, y, fig_name, test_size=1/5, epochs=100, learning_rate=0.01, momentum=0.9):
    print("Calculating results for "+fig_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    y_train = y_train.T
    y_test = y_test.T

    mlp = MLP(layers, X_train, y_train, epochs, learning_rate, momentum)
    mlp_2 = MLP(layers_2, X_train, y_train, epochs, learning_rate, momentum)

    loss = mlp.train()
    loss_2 = mlp_2.train()

    # Accuracy
    predictions = mlp.predict(X_test)
    predictions_2 = mlp_2.predict(X_test)

    mse_test = mlp.compute_loss(y_test, predictions)
    mse_test_2 = mlp_2.compute_loss(y_test, predictions_2)

    print(f"Mean Squared error: {mse_test:.4f}")
    print(f"Mean Squared error 2: {mse_test_2:.4f}")
    
    plt.title('Learning rate='+str(round(learning_rate, 3))+' Momentum='+str(round(momentum,2)))
    plt.plot(loss, 'r-', linewidth=2, label='HiddenLayers=2')
    plt.plot(loss_2, 'g-', linewidth=2, label='HiddenLayers=1')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='lower left')
    plt.ylim(ymin = 0)
    plt.savefig('Projeto 01 - MLP/'+fig_name+'.png')
    plt.close()
    print("\n")   

#Data pre-proccessing
data = pd.read_csv('Projeto 01 - MLP\default_features_1059_tracks.txt', delimiter=',', header=None)
X = data.iloc[:, :-2].values  # Features
y = data.iloc[:, -2:].values  # Lat&Long

# Input normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# X_scaled = (X - np.average(X)) / (np.std(X))

epochs = 500
learning_rate = 0.005
momentum=0.9

# Training
layers = [68, 64, 32, 2]

layers_2 = [68, 32, 2]

# Epochs = 100, Learning rate = 0.01, Momentum = 0.9, TestSize = 1/5
plot_results(layers, layers_2, X_scaled, y, "Music - Padrão")

# Epochs = 100, Learning rate = 0.01, Momentum = 0.5, TestSize = 1/5
plot_results(layers, layers_2, X_scaled, y, "Music - Momentum Variation", momentum=0.5)

# Epochs = 500, Learning rate = 0.01, Momentum = 0.9, TestSize = 1/5
plot_results(layers, layers_2, X_scaled, y, "Music - Epochs Variation", epochs=500)

# Epochs = 100, Learning rate = 0.1, Momentum = 0.9, TestSize = 1/5
plot_results(layers, layers_2, X_scaled, y, "Music - Learning Rate Variation", learning_rate=0.1)

# Epochs = 100, Learning rate = 0.01, Momentum = 0.9, TestSize = 1/3
plot_results(layers, layers_2, X_scaled, y, "Music - Test Size Variation", test_size=1/3)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, layers, X, y, epochs, learning_rate, momentum, verbose=False):
        self.layers = layers
        self.weights, self.biases = self.initialize_weights()
        self.X = X
        self.y = y.T
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.verbose = verbose

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
    
    # Softmax (for classification)
    def softmax(self, l):
        exp_l = np.exp(l - np.max(l, axis=0, keepdims=True))  # Avoids numeric overflow
        return exp_l / np.sum(exp_l, axis=0, keepdims=True)
    
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
        A_output = self.softmax(L_output)

        activations.append(A_output)
        return L_values, activations
    
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[1]
        log_likelihood = -np.log(y_pred[y_true.argmax(axis=0), np.arange(m)])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward_propagation(self, L_values, activations):
        m = self.X.shape[0]
        dW, db = [], []
        dA_prev = -(self.y - activations[-1])  # SoftMax gradient

        # Momentum initial term
        momentum_dW = [np.zeros_like(w) for w in self.weights]
        momentum_db = [np.zeros_like(b) for b in self.biases]

        # Backpropagation for output layer (softmax)
        dL = dA_prev  # SoftMax derivative: dL = A - y_true
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
        accuracies = []
        for epoch in range(self.epochs):
            # Forward propagation
            L_values, activations = self.forward_propagation(self.X)
            
            # Loss 
            loss = self.compute_loss(self.y, activations[-1])
            accuracy = np.mean(activations[-1].argmax(axis=0) == self.y.T.argmax(axis=1))
            losses.append(loss)
            accuracies.append(accuracy)

            # Backpropagation
            self.backward_propagation(L_values, activations)
            
            if epoch % 100 == 0 and self.verbose:
                print(f"Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}")
        return losses, accuracies

    def predict(self, X):
        _, activations = self.forward_propagation(X)
        return activations[-1].argmax(axis=0)
    
def plot_results(layers, layers_2, X, y, fig_name, test_size=1/5, epochs=1000, learning_rate=0.01, momentum=0.9):
    print("Calculating results for "+fig_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    mlp = MLP(layers, X_train, y_train, epochs, learning_rate, momentum)
    mlp_2 = MLP(layers_2, X_train, y_train, epochs, learning_rate, momentum)

    loss, accuracy = mlp.train()
    loss_2, accuracy_2 = mlp_2.train()

    # Accuracy
    predictions = mlp.predict(X_test)
    predictions_2 = mlp_2.predict(X_test)

    score = np.mean(predictions == y_test.argmax(axis=1))
    score_2 = np.mean(predictions_2 == y_test.argmax(axis=1))
    print(f"Two layer MLP Accuracy: {score * 100:.2f}%")
    print(f"One layer MLP Accuracy: {score_2 * 100:.2f}%")
    
    plt.title('Learning rate='+str(round(learning_rate, 3))+' Momentum='+str(round(momentum,2)))
    plt.plot(accuracy, 'r-', linewidth=2, label='Hidden Layers=2')
    plt.plot(accuracy_2, 'g-', linewidth=2, label='Hidden Layers=1')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    plt.ylim(ymin = 0, ymax = 1.1)
    plt.savefig('Projeto 01 - MLP/'+fig_name+'.png')
    plt.close()
    print("\n")

#Data pre-proccessing
wine = fetch_ucirepo(id=109)
X = pd.DataFrame(wine.data.features).to_numpy()
y = pd.DataFrame(wine.data.targets).to_numpy()
y = y.T - 1
y = y[0]

# One-hot encoding for y
y_onehot = np.eye(len(np.unique(y)))[y]

# Input normalization
scaler = StandardScaler()
# X_scaled = (X - np.average(X)) / (np.std(X))
X_scaled = scaler.fit_transform(X)

layers = [X.shape[1], 13, 6, 3]
layers_2 = [X.shape[1], 6, 3]

# Epochs = 1000, Learning rate = 0.01, Momentum = 0.9, TestSize = 1/5
plot_results(layers, layers_2, X_scaled, y_onehot, "Wine - Padrão")

# Epochs = 1000, Learning rate = 0.01, Momentum = 0.5, TestSize = 1/5
plot_results(layers, layers_2, X_scaled, y_onehot, "Wine - Momentum Variation", momentum=0.5)

# Epochs = 5000, Learning rate = 0.01, Momentum = 0.9, TestSize = 1/5
plot_results(layers, layers_2, X_scaled, y_onehot, "Wine - Epochs Variation", epochs=5000)

# Epochs = 1000, Learning rate = 0.1, Momentum = 0.9, TestSize = 1/5
plot_results(layers, layers_2, X_scaled, y_onehot, "Wine - Learning Rate Variation", learning_rate=0.1)

# Epochs = 1000, Learning rate = 0.01, Momentum = 0.9, TestSize = 1/3
plot_results(layers, layers_2, X_scaled, y_onehot, "Wine - Test Size Variation", test_size=1/3)

# Epochs = 1000, Learning rate = 0.01, Momentum = 0.9, TestSize = 1/3
plot_results(layers, layers_2, X_scaled, y_onehot, "Wine - All Variation", test_size=1/3,  momentum=0.5, learning_rate=0.1, epochs=5000)

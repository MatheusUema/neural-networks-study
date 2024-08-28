import numpy as np
import matplotlib.pyplot as plt

# "Y" Symbol with label 1
y_symbol = (1, np.array([
    [ 1, -1, -1, -1,  1],
    [ 1, -1, -1, -1,  1],
    [-1,  1,  1,  1, -1],
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1]
]))

# Inverted "Y" Symbol with label -1
y_inverted_symbol = (-1, np.array([
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1],
    [-1,  1,  1,  1, -1],
    [ 1, -1, -1, -1,  1],
    [ 1, -1, -1, -1,  1]
]))

# Function to convert matrix to array
def flatten_symbol(symbol):
    return symbol.flatten()

# Create training data
def create_data(data_tuples):
    X = []
    y = np.array([])
    for label, data in data_tuples:
        X.append(flatten_symbol(data))
        y = np.append(y, label)
    
    X = np.array(X)
    return X, y

# Training data
X_train, y_train = create_data([y_symbol, 
                                y_inverted_symbol])

# Adaline class
class Adaline:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size)
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return x  # Linear function activation

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.0, 1, -1)

    def train(self, X, y):
        for epoch in range(self.epochs):
            outputs = self.activation_function(np.dot(X, self.weights) + self.bias)
            errors = y - outputs
            self.weights += self.learning_rate * np.dot(X.T, errors)
            self.bias += self.learning_rate * errors.sum()
            
            # Calculating Mean Squared error
            mse = (errors**2).mean()
            print(f"Epoch {epoch+1}/{self.epochs} - MSE: {mse:.4f}")

# Initializing model
adaline = Adaline(input_size=25, learning_rate=0.001, epochs=20)

# Model training
adaline.train(X_train, y_train)

def model_predict(adaline, X_train, y_train):
    score = 0
    for idx, sample in enumerate(X_train):
        prediction = adaline.predict(sample)
        symbol = 'Y' if prediction == 1 else 'Inverted Y'
        print(f"Sample {idx+1}: Prediction = {symbol}, Real = {'Y' if y_train[idx] == 1 else 'Inverted Y'}")
        if(prediction == y_train[idx]):
            score +=1
    prediction_score = 100*score/len(X_train)
    print(f"Total score prediction: {format(prediction_score, ".2f")}%")

# Testing with the training data
# model_predict(adaline, X_train, y_train)

#Defining test data
# Test - "Y" Symbol + noise
test_y_symbol_1 = (1, np.array([
    [ 1, -1,  1, -1,  1],
    [-1,  1,  1,  1, -1],
    [-1, -1,  1, -1, -1],
    [-1,  1,  1,  1, -1],
    [ 1,  1,  1,  1,  1]
]))

test_y_symbol_2 = (1, np.array([
    [ 1, -1, -1, -1,  1],
    [-1,  1, -1,  1, -1],
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1],
    [ 1, -1,  1, -1, -1]
]))


test_y_symbol_3 = (1, np.array([
    [ 1, -1, -1, -1,  1],
    [-1,  1, -1,  1, -1],
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1],
    [ 1, -1,  1, -1,  1]
]))

test_y_symbol_4 = (1, np.array([
    [ 1,  1,  1,  1,  1],
    [-1,  1, -1,  1, -1],
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1]
]))

test_y_symbol_5 = (1, np.array([
    [ 1, -1, -1, -1,  1],
    [-1,  1, -1,  1, -1],
    [-1, -1,  1, -1, -1],
    [ 1, -1,  1, -1,  1],
    [-1, -1,  1,  1, -1]
]))

test_y_symbol_6 = (1, np.array([
    [ 1, -1, -1, -1,  1],
    [-1,  1, -1,  1, -1],
    [-1, -1,  1, -1, -1],
    [-1,  1,  1,  1, -1],
    [ 1, -1,  1, -1,  1]
]))


# Test - Inverted "Y" Symbol + noise
test_y_inverted_symbol_1 = (-1, np.array([
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1],
    [-1,  1, -1,  1, -1],
    [ 1, -1,  1, -1,  1]
]))

test_y_inverted_symbol_2 = (-1, np.array([
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1],
    [-1, -1,  1, -1, -1],
    [-1,  1,  1,  1, -1],
    [ 1,  1,  1,  1,  1]
]))

test_y_inverted_symbol_3 = (-1, np.array([
    [-1, -1,  1,  1, -1],
    [-1, -1,  1, -1, -1],
    [-1, -1,  1,  1, -1],
    [-1,  1, -1,  1, -1],
    [ 1, -1, -1, -1,  1]
]))

test_y_inverted_symbol_4 = (-1, np.array([
    [ 1, -1,  1, -1, -1],
    [ 1, -1,  1, -1, -1],
    [ 1, -1,  1, -1, -1],
    [-1,  1, -1,  1, -1],
    [ 1, -1,  1, -1,  1]
]))

test_y_inverted_symbol_5 = (-1, np.array([
    [-1, -1,  1, -1,  1],
    [-1,  1,  1, -1, -1],
    [-1, -1,  1, -1, -1],
    [-1,  1, -1,  1, -1],
    [ 1, -1, -1, -1,  1]
]))

test_y_inverted_symbol_6 = (-1, np.array([
    [-1, -1,  1, -1,  1],
    [ 1, -1,  1, -1,  1],
    [-1, -1,  1, -1, -1],
    [-1,  1, -1,  1, -1],
    [ 1, -1,  1, -1,  1]
]))

# Test data
X_test, y_test = create_data([test_y_symbol_1, 
                              test_y_symbol_2, 
                              test_y_symbol_3,
                              test_y_symbol_4,
                              test_y_symbol_5,
                              test_y_symbol_6,
                              test_y_inverted_symbol_1, 
                              test_y_inverted_symbol_2, 
                              test_y_inverted_symbol_3,
                              test_y_inverted_symbol_4,
                              test_y_inverted_symbol_5,
                              test_y_inverted_symbol_6])

model_predict(adaline, X_test, y_test)

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_der(x):
	return x*(1.0 - x)

class MLP:
    def __init__(self, inputs, learning_rate, epochs):
        self.inputs = inputs
        self.l=len(self.inputs) # 4
        self.li=len(self.inputs[0]) # 2
        self.epochs = epochs
        self.learning_rate = learning_rate

        #randomizing weights for hidden layer and for output layer
        self.wi=np.random.random((self.li, self.l))
        self.wh=np.random.random((self.l, 1))

    def predict(self, inp):
        s1=sigmoid(np.dot(inp, self.wi))
        s2=sigmoid(np.dot(s1, self.wh))

        return s2

    def train(self, inputs, outputs):
        cost = []
        for i in range(self.epochs):
            l0=inputs
            l1=sigmoid(np.dot(l0, self.wi))
            l2=sigmoid(np.dot(l1, self.wh))

            l2_err=outputs - l2
            l2_delta = np.multiply(l2_err, sigmoid_der(l2))

            l1_err=np.dot(l2_delta, self.wh.T)
            l1_delta=np.multiply(l1_err, sigmoid_der(l1))

            cost.append(np.mean(np.abs(l2_delta)))

            self.wh+=np.dot(l1.T, l2_delta)*self.learning_rate
            self.wi+=np.dot(l0.T, l1_delta)*self.learning_rate
        return cost

inputs=np.array([[0,0], 
                 [0,1], 
                 [1,0], 
                 [1,1]])
outputs=np.array([ [0],
                   [1],
                   [1],
                   [0]])

n=MLP(inputs, 0.5, 10000)
print(n.predict(inputs))
costs = n.train(inputs, outputs)
print(n.predict(inputs))

plt.plot(costs)
plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# IdentityMatrix
matrix_size = 11
hidden_nn_num = np.floor(np.log2(matrix_size)).astype(int)
input_data = torch.eye(matrix_size)

# AutoEncoder class
class AutoEncoder(nn.Module):
    def __init__(self, matrix_size, hidden_nn_num):
        super(AutoEncoder, self).__init__()

        # Coder layer 
        self.encoder = nn.Sequential(
            nn.Linear(matrix_size, hidden_nn_num),
            nn.ReLU()
        )

        # Decoder Layer
        self.decoder = nn.Sequential(
            nn.Linear(hidden_nn_num, matrix_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = AutoEncoder(matrix_size, hidden_nn_num)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# Training
epochs = 30000
costs = []
for epoch in range(epochs):
    # Forward pass
    outputs = model(input_data)
    loss = criterion(outputs, input_data)
    costs.append(loss.item())
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Plot
with torch.no_grad():
    reconstructed = model(input_data)
    print("Entrada Original:\n", input_data)
    print("Saída Reconstruída:\n", reconstructed)

plt.plot(costs)
plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

# ----------------------------
# 1️⃣ Generar datos
# ----------------------------
X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=42)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# ----------------------------
# 2️⃣ Definir modelo tipo KAN simplificado
# (Lineal para frontera recta)
# ----------------------------
class SimpleKAN(nn.Module):
    def __init__(self):
        super(SimpleKAN, self).__init__()
        self.linear = nn.Linear(2, 1)  # frontera lineal

    def forward(self, x):
        return self.linear(x)

model = SimpleKAN()

# ----------------------------
# 3️⃣ Función de pérdida y optimizador
# ----------------------------
criterion = nn.BCEWithLogitsLoss()  # entropía cruzada binaria
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ----------------------------
# 4️⃣ Entrenamiento
# ----------------------------
epochs = 1000

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ----------------------------
# 5️⃣ Graficar resultados
# ----------------------------
plt.figure(figsize=(8,6))

# Graficar puntos
plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap='bwr', alpha=0.7)

# Obtener parámetros de la recta
w = model.linear.weight.data.numpy()[0]
b = model.linear.bias.data.numpy()[0]

# Línea de decisión: w1*x1 + w2*x2 + b = 0
x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
y_vals = -(w[0]*x_vals + b) / w[1]

plt.plot(x_vals, y_vals, 'k--', label="Frontera de decisión")

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Clasificación binaria con KAN (Entropía Cruzada)")
plt.legend()
plt.show()


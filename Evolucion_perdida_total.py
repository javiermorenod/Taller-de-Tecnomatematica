# ------------------------- 
# 1. Importar librerías
# -------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ------------------------- 
# 2. Configuración del dispositivo
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------- 
# 3. Cargar y transformar datos MNIST
# -------------------------
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

# ------------------------- 
# 4. Modelo: Autoencoder + clasificador softmax
# -------------------------
class AEWithSoftmaxHead(nn.Module):
    def __init__(self, latent_dim=16, n_classes=10):
        super().__init__()

        # Encoder: Mapeo de la imagen a espacio latente
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),  # Entrada: 784 (28x28) a 128
            nn.ReLU(),
            nn.Linear(128, latent_dim)  # Codificación a dimensión latente
        )

        # Decoder: Reconstrucción de la imagen desde el espacio latente
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),  # Latent dim a 128
            nn.ReLU(),
            nn.Linear(128, 784),  # Reconstrucción a 784 dimensiones
            nn.Sigmoid()  # Salida en [0, 1]
        )

        # Head: logits = zW + b ; y_hat = softmax(logits)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, x):
        x = x.view(-1, 784)  # Aplana la imagen
        z = self.encoder(x)  # Codificación
        x_hat = self.decoder(z)  # Reconstrucción
        logits = self.classifier(z)  # Predicción
        y_hat = F.softmax(logits, dim=1)  # Probabilidades de clase (softmax)
        return z, x_hat, logits, y_hat

model = AEWithSoftmaxHead(latent_dim=16, n_classes=10).to(device)

# ------------------------- 
# 5. Pérdidas y optimizador
# -------------------------
# Pérdida de reconstrucción (como en el ejemplo)
recon_criterion = nn.BCELoss()

# Pérdida de clasificación: MSE sobre one-hot vs softmax
cls_criterion = nn.MSELoss()

# Peso de la reconstrucción:
LAMBDA_RECON = 1.0  # Controla la importancia de la reconstrucción frente a la clasificación

# Usamos optimizador SGD
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# ------------------------- 
# 6. Entrenamiento
# -------------------------
epochs = 10

loss_total_hist, loss_recon_hist, loss_cls_hist, acc_hist = [], [], [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_cls = 0.0
    correct = 0
    n = 0

    for images, labels in train_loader:
        images = images.to(device)  # [B,1,28,28]
        labels = labels.to(device)  # [B]

        # One-hot: [B,10]
        y = F.one_hot(labels, num_classes=10).float()

        # Forward pass
        z, x_hat, logits, y_hat = model(images)

        # Pérdida de clasificación (MSE)
        loss_cls = cls_criterion(y_hat, y)

        # Pérdida de reconstrucción (BCE)
        x_flat = images.view(-1, 784)  # Aplanar las imágenes
        loss_recon = recon_criterion(x_hat, x_flat)

        # Pérdida total
        loss = loss_cls + LAMBDA_RECON * loss_recon

        # Optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Métricas
        total_loss += loss.item()
        total_recon += loss_recon.item()
        total_cls += loss_cls.item()

        # Cálculo de exactitud
        preds = torch.argmax(y_hat, dim=1)
        correct += (preds == labels).sum().item()
        n += labels.size(0)

    # Promediar las pérdidas y la exactitud
    avg_loss = total_loss / len(train_loader)
    avg_recon = total_recon / len(train_loader)
    avg_cls = total_cls / len(train_loader)
    acc = correct / n

    loss_total_hist.append(avg_loss)
    loss_recon_hist.append(avg_recon)
    loss_cls_hist.append(avg_cls)
    acc_hist.append(acc)

    print(f"Época [{epoch+1}/{epochs}] | loss={avg_loss:.4f} | cls(MSE)={avg_cls:.4f} | recon(BCE)={avg_recon:.4f} | acc={acc*100:.2f}%")

# ------------------------- 
# 7. Curvas de pérdidas y precisión
# -------------------------
plt.figure()
plt.plot(loss_total_hist, label="Loss total")
plt.plot(loss_cls_hist, label="Loss clasificador (MSE)")
plt.plot(loss_recon_hist, label=f"Loss reconstrucción (BCE)*{LAMBDA_RECON}")
plt.title("Pérdidas por época")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(acc_hist)
plt.title("Precisión por época (train)")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.grid(True)
plt.show()

# ------------------------- 
# 8. Visualización: Ejemplo de imagen por clase (original, reconstrucción, y predicción)
# -------------------------
model.eval()

# Guardar un ejemplo por clase
examples = {i: None for i in range(10)}
example_labels = {}

for img, label in train_dataset:
    if examples[label] is None:
        examples[label] = img
    if all(v is not None for v in examples.values()):
        break

fig, axes = plt.subplots(10, 3, figsize=(6.5, 20))

with torch.no_grad():
    for i in range(10):
        original = examples[i].to(device).view(1, -1)  # [1,784]
        z = model.encoder(original)
        recon = model.decoder(z)
        logits = model.classifier(z)
        y_hat = F.softmax(logits, dim=1).cpu().numpy().ravel()

        axes[i, 0].imshow(original.cpu().view(28, 28), cmap='gray')
        axes[i, 0].set_title(f"Original {i}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(recon.cpu().view(28, 28), cmap='gray')
        axes[i, 1].set_title("Reconstrucción")
        axes[i, 1].axis('off')

        pred = int(y_hat.argmax())
        axes[i, 2].bar(range(10), y_hat)
        axes[i, 2].set_title(f"y_hat (pred={pred})")
        axes[i, 2].set_xticks(range(10))
        axes[i, 2].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# ------------------------- 
# 9. (Opcional) Ver W y b del head: softmax(zW + b)
# -------------------------
W = model.classifier.weight.detach().cpu()  # [10, latent_dim]
b = model.classifier.bias.detach().cpu()  # [10]
print("\nW shape:", tuple(W.shape))
# %%
print("b shape:", tuple(b.shape))

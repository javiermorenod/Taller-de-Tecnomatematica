import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import imageio

# -----------------------------
# Parámetros
# -----------------------------
batch_size = 128
num_epochs = 20
latent_dim = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Carga y preprocesado de datos
# -----------------------------
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# Definición del Autoencoder
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

model = Autoencoder().to(device)

# -----------------------------
# Pérdida y optimizador
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# Entrenamiento
# -----------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, _ in train_loader:
        images = images.view(images.size(0), -1).to(device)

        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# -----------------------------
# Seleccionar un 1 y un 8
# -----------------------------
model.eval()

img_1 = None
img_8 = None

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        for i in range(len(labels)):
            if labels[i].item() == 1 and img_1 is None:
                img_1 = images[i].view(1, -1)
            if labels[i].item() == 8 and img_8 is None:
                img_8 = images[i].view(1, -1)

            if img_1 is not None and img_8 is not None:
                break
        if img_1 is not None and img_8 is not None:
            break

# -----------------------------
# Codificación latente
# -----------------------------
with torch.no_grad():
    z_1 = model.encoder(img_1)
    z_8 = model.encoder(img_8)

# -----------------------------
# Interpolación latente y GIF
# -----------------------------
num_frames = 30
frames = []

with torch.no_grad():
    for t in np.linspace(0, 1, num_frames):
        z_interp = (1 - t) * z_1 + t * z_8
        recon = model.decoder(z_interp)
        img = recon.view(28, 28).cpu().numpy()

        img_uint8 = (img * 255).astype(np.uint8)
        frames.append(img_uint8)

imageio.mimsave("latent_1_to_8.gif", frames, fps=6)

print("GIF generado: latent_1_to_8.gif")
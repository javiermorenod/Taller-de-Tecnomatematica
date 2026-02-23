import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------
# 1️⃣ Transformación
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

# ---------------------------
# 2️⃣ Cargar MNIST
# ---------------------------
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ---------------------------
# 3️⃣ Definición del Autoencoder
# ---------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

model = Autoencoder()

# ---------------------------
# 4️⃣ Pérdida y optimizador
# ---------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# 5️⃣ Entrenamiento
# ---------------------------
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, _ in train_loader:
        images = images.view(images.size(0), -1)
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()

# ---------------------------
# 6️⃣ Elegir imágenes 2 y 6
# ---------------------------
img2, _ = next((img, lbl) for img, lbl in test_dataset if lbl==2)
img6, _ = next((img, lbl) for img, lbl in test_dataset if lbl==6)

img2_input = img2.view(1, -1)
img6_input = img6.view(1, -1)

with torch.no_grad():
    _, z2 = model(img2_input)
    _, z6 = model(img6_input)

# ---------------------------
# 7️⃣ Interpolación y GIF
# ---------------------------
steps = 30  # número de frames
frames = []

plt.figure(figsize=(4,4))

for alpha in torch.linspace(0, 1, steps):
    z_interp = (1-alpha)*z2 + alpha*z6
    recon = model.decoder(z_interp).view(28,28).detach().numpy()
    
    # Mostrar el frame en matplotlib
    plt.clf()
    plt.imshow(recon, cmap='gray')
    plt.axis('off')
    plt.title(f"Interpolación: alpha={alpha:.2f}")
    plt.pause(0.05)  # pausa corta para animación
    
    # Convertir a PIL para GIF
    im = Image.fromarray((recon*255).astype('uint8')).resize((140,140))
    frames.append(im)

plt.close()

# Guardar GIF
frames[0].save(
    '2_to_6_interpolation.gif',
    save_all=True,
    append_images=frames[1:],
    duration=150,  # velocidad del GIF
    loop=0
)

print("GIF generado: 2_to_6_interpolation.gif")
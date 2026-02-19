
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


batch_size = 128

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transform)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              download=True,
                              transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Autoencoder2D(nn.Module):
    def __init__(self):
        super(Autoencoder2D, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 2)   # ← dimensión 2
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder2D().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

num_epochs = 20

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

model.eval()

digit_examples = {}
reconstructions = {}

with torch.no_grad():
    for images, labels in test_loader:
        for i in range(len(labels)):
            label = labels[i].item()
            
            if label not in digit_examples:
                digit_examples[label] = images[i]
                
                img = images[i].view(1, -1).to(device)
                reconstructed = model(img)
                reconstructions[label] = reconstructed.cpu().view(28, 28)
            
            if len(digit_examples) == 10:
                break
        if len(digit_examples) == 10:
            break

# Plot
plt.figure(figsize=(12, 4))

for digit in range(10):
    # Original
    plt.subplot(2, 10, digit+1)
    plt.imshow(digit_examples[digit].view(28, 28), cmap='gray')
    plt.axis('off')
    
    # Reconstrucción
    plt.subplot(2, 10, digit+11)
    plt.imshow(reconstructions[digit], cmap='gray')
    plt.axis('off')

plt.show()




model.eval()

latent_points = []
labels_list = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(images.size(0), -1).to(device)
        z = model.encoder(images)
        latent_points.append(z.cpu())
        labels_list.append(labels)

latent_points = torch.cat(latent_points)
labels_list = torch.cat(labels_list)

plt.figure(figsize=(8,6))
plt.scatter(latent_points[:,0],
            latent_points[:,1],
            c=labels_list,
            cmap='tab10',
            s=5)

plt.colorbar()
plt.title("Espacio Latente 2D")
plt.xlabel("z1")
plt.ylabel("z2")
plt.show()



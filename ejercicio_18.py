import numpy as np
import matplotlib.pyplot as plt
import imageio
import io

# Datos
X = np.array([0, 1, 3, 4])
Y = np.array([0, 2, 2, 5])

# Hiperparámetros
eta = 0.05
epochs = 20

# Inicialización
w, b = 0.0, 0.0
losses = []
frames = []

# Entrenamiento
for epoch in range(epochs):
    y_hat = w * X + b
    errors = y_hat - Y

    # pérdida
    L = 0.5 * np.sum(errors**2)
    losses.append(L)

    # gradientes
    dw = np.sum(errors * X)
    db = np.sum(errors)

    # actualización
    w -= eta * dw
    b -= eta * db

    # Crear la figura para el GIF
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # subplot 1: recta
    xline = np.linspace(0, 5, 100)
    axs[0].scatter(X, Y)
    axs[0].plot(xline, w*xline+b)
    axs[0].set_title(f"Epoch {epoch}")
    axs[0].set_xlim(0, 5)
    axs[0].set_ylim(-1, 6)

    # subplot 2: pérdida
    axs[1].plot(losses)
    axs[1].set_title("Loss")
    axs[1].set_yscale("log")

    plt.tight_layout()

    # Usar buffer en memoria
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    frames.append(imageio.imread(buf))
    buf.close()

    plt.close()

# Guardar GIF
imageio.mimsave("ejercicio_18.gif", frames, fps=10)

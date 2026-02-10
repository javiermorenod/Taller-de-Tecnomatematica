import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io

# 1. DATOS: Cambiamos Y a valores binarios (0 o 1)
# Imagina que 0 es "Suspendido" y 1 es "Aprobado"
X = np.array([0, 1, 3, 4])
Y = np.array([0, 0, 1, 1])

# Hiperparámetros (Ajustamos eta y epochs para que converja visiblemente)
eta = 0.5
epochs = 100

# Inicialización
w, b = 0.0, 0.0
losses = []
frames = []


# Definición de la función Sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Entrenamiento
for epoch in range(epochs):

    # 2. MODELO: Regresión lineal + Activación Sigmoide
    z = w * X + b
    y_hat = sigmoid(z)  # Predicciones (probabilidad entre 0 y 1)

    # Cálculo del error (predicción - real)
    errors = y_hat - Y

    # 3. PÉRDIDA: Binary Cross Entropy (Log Loss)
    # Añadimos 1e-15 para evitar log(0)
    L = -np.mean(Y * np.log(y_hat + 1e-15) + (1 - Y)
                 * np.log(1 - y_hat + 1e-15))
    losses.append(L)

    # GRADIENTES
    # Matemáticamente, para sigmoide + log loss, la derivada simplificada
    # es idéntica a la de regresión lineal: (y_hat - y) * x
    dw = np.dot(errors, X) / len(X)  # Promedio
    db = np.sum(errors) / len(X)    # Promedio

    # Actualización
    w -= eta * dw
    b -= eta * db

    # --- VISUALIZACIÓN ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Subplot 1: La curva sigmoide (Curva S)
    xline = np.linspace(-1, 5, 100)
    zline = w * xline + b
    yline = sigmoid(zline)

    axs[0].scatter(X, Y, c=Y, cmap='bwr', s=100, edgecolor='k', label='Datos')
    axs[0].plot(xline, yline, color='green', label='Probabilidad (Sigmoide)')

    # Dibujar la frontera de decisión (donde probabilidad = 0.5)
    axs[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axs[0].set_title(f"Epoch {epoch} | w: {w:.2f}, b: {b:.2f}")
    axs[0].set_ylim(-0.1, 1.1)
    axs[0].legend()

    # Subplot 2: Pérdida
    axs[1].plot(losses, color='red')
    axs[1].set_title("Log Loss (Coste)")
    axs[1].set_xlabel("Epoch")

    plt.tight_layout()

    # Guardar frame
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    frames.append(imageio.imread(buf))
    buf.close()
    plt.close()

# Guardar GIF
imageio.mimsave("ejercicio_clasificacion.gif", frames, fps=10)

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import shutil

# -----------------------
# Parámetros
# -----------------------
np.random.seed(0)
N_neurons = 5
lr = 0.01
epochs = 2000
lambda_bc = 10.0

# Dominio
x = np.linspace(0, 3, 100)

# -----------------------
# Inicialización
# -----------------------
a = np.random.randn(N_neurons)
b = np.random.randn(N_neurons)
c = np.random.randn(N_neurons)
d = np.random.randn()

# -----------------------
# Funciones de la Red
# -----------------------


def y(x):
    # Forward pass: sum(a * tanh(b*x + c)) + d
    return np.sum(a[:, None] * np.tanh(b[:, None]*x + c[:, None]), axis=0) + d


def y_xx(x):
    # Segunda derivada analítica de la red
    t = np.tanh(b[:, None]*x + c[:, None])
    return np.sum(-2*a[:, None]*b[:, None]**2 * t * (1 - t**2), axis=0)


# -----------------------
# Entrenamiento
# -----------------------
frames = []
loss_history = []

os.makedirs("frames_temp", exist_ok=True)

for epoch in range(epochs):

    # 1. Cálculo de Pérdida (Loss)
    yxx = y_xx(x)
    loss_phys = np.mean(yxx**2)

    # CORRECCIÓN: Usamos .item() para que loss_bc sea un número, no un array
    # y(0) devuelve un array de forma (1,), al sumar con y(3) sigue siendo array.
    val_0 = y(0)
    val_3 = y(3)
    loss_bc = (val_0 - 2)**2 + (val_3 - 5)**2

    # Forzamos a que 'loss' sea un escalar puro usando .item() si es necesario
    # Nota: np.mean devuelve escalar, pero loss_bc podría ser array(1,)
    loss = loss_phys + lambda_bc * loss_bc

    # Si loss es un array, extraemos el valor. Si ya es escalar, esto no afecta.
    if isinstance(loss, np.ndarray):
        loss = loss.item()

    loss_history.append(loss)

    # 2. Gradientes (Diferencias finitas manuales)
    eps = 1e-5

    # Actualización de pesos a, b, c
    for i in range(N_neurons):
        for param_array in [a, b, c]:
            original_val = param_array[i]

            # Perturbación positiva
            param_array[i] = original_val + eps
            # Calculamos lp y forzamos a escalar con .item()
            val_lp_0 = y(0)
            val_lp_3 = y(3)
            lp = np.mean(y_xx(x)**2) + lambda_bc * \
                ((val_lp_0-2)**2 + (val_lp_3-5)**2)
            if isinstance(lp, np.ndarray):
                lp = lp.item()

            # Perturbación negativa
            param_array[i] = original_val - eps
            val_lm_0 = y(0)
            val_lm_3 = y(3)
            lm = np.mean(y_xx(x)**2) + lambda_bc * \
                ((val_lm_0-2)**2 + (val_lm_3-5)**2)
            if isinstance(lm, np.ndarray):
                lm = lm.item()

            # Restaurar valor
            param_array[i] = original_val

            # Actualizar (Descenso de gradiente)
            grad = (lp - lm) / (2 * eps)
            param_array[i] -= lr * grad

    # Actualización del sesgo d
    d_orig = d
    d = d_orig + eps

    # Calculamos lp para d (mismo proceso de convertir a item)
    val_lp_0 = y(0)
    val_lp_3 = y(3)
    lp = np.mean(y_xx(x)**2) + lambda_bc*((val_lp_0-2)**2 + (val_lp_3-5)**2)
    if isinstance(lp, np.ndarray):
        lp = lp.item()

    d = d_orig - eps

    # Calculamos lm para d
    val_lm_0 = y(0)
    val_lm_3 = y(3)
    lm = np.mean(y_xx(x)**2) + lambda_bc*((val_lm_0-2)**2 + (val_lm_3-5)**2)
    if isinstance(lm, np.ndarray):
        lm = lm.item()

    d = d_orig
    d -= lr * (lp - lm) / (2 * eps)

    # -----------------------
    # Visualización Combinada
    # -----------------------
    if epoch % 50 == 0:
        # Creamos una figura con 2 columnas
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # --- Gráfico Izquierdo: Solución ---
        # .ravel() aplana por seguridad
        ax1.plot(x, y(x).ravel(), 'r-', label="PINN")
        ax1.plot(x, x+2, "k--", label="Exacta", alpha=0.6)
        ax1.set_ylim(1.5, 5.5)
        ax1.set_title(f"Solución (Epoch {epoch})")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # --- Gráfico Derecho: Pérdida ---
        ax2.semilogy(loss_history, 'b-')
        ax2.set_title("Evolución del Loss")
        ax2.set_xlabel("Iteraciones")
        ax2.grid(True, which="both", alpha=0.3)

        plt.tight_layout()

        filename = f"frames_temp/frame_{epoch}.png"
        plt.savefig(filename)
        plt.close()

        frames.append(imageio.imread(filename))

imageio.mimsave("primera_entrega_caso_2.gif", frames, fps=10)

shutil.rmtree("frames_temp")

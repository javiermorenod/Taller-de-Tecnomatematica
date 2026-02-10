import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio
import io

# --- 1. DATOS (Del primer código) ---
X = np.array([[2, 4], [3, 6], [2, 0], [4, -5]])
y = np.array([1, 1, 0, 0])

# --- 2. HIPERPARÁMETROS ---
tasa_aprendizaje = 0.00000000000001
n_epochs = 10

# Inicialización
w = np.zeros(2)
b = 0.0
errores_historico = []
frames = []

# Configuración del gráfico (Limites y Colores)
xlim = (-10, 10)
ylim = (-10, 10)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])

# --- 3. ENTRENAMIENTO (Estructura tipo "imageio") ---
for epoch in range(n_epochs):
    errores_en_epoca = 0

    # --- Lógica de Entrenamiento del Perceptrón ---
    # (Iteramos muestra por muestra)
    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        prediccion = 1 if z >= 0 else 0
        error = y[i] - prediccion

        if error != 0:
            w += tasa_aprendizaje * error * X[i]
            b += tasa_aprendizaje * error
            errores_en_epoca += 1

    errores_historico.append(errores_en_epoca)

    # --- 4. GENERACIÓN DE LA IMAGEN (Frame) ---
    # Creamos la figura (2 subplots para parecerse al estilo del código 2)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # --- Subplot 1: Frontera de decisión ---
    ax = axs[0]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"Época {epoch} | Errores: {errores_en_epoca}")

    # a. Dibujar fondo (Malla de decisión)
    xx, yy = np.meshgrid(np.arange(xlim[0], xlim[1], 0.05),
                         np.arange(ylim[0], ylim[1], 0.05))
    Z_grid = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z_grid = np.array([1 if z >= 0 else 0 for z in Z_grid]).reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z_grid, cmap=cmap_light, shading='auto', alpha=0.5)

    # b. Dibujar Puntos
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1],
               color='red', s=100, edgecolors='k', label='0')
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1],
               color='green', s=100, edgecolors='k', label='1')

    # c. Dibujar la línea recta (Frontera)
    x_vals = np.array(xlim)
    if abs(w[1]) > 1e-5:  # Evitar división por cero
        y_vals = -(w[0] * x_vals + b) / w[1]
        ax.plot(x_vals, y_vals, 'k--', linewidth=2)
    elif abs(w[0]) > 1e-5:  # Caso línea vertical
        x_vertical = -b / w[0]
        ax.plot([x_vertical, x_vertical], ylim, 'k--', linewidth=2)

    # --- Subplot 2: Historial de Errores ---
    ax2 = axs[1]
    ax2.plot(range(len(errores_historico)),
             errores_historico, marker='o', linestyle='-')
    ax2.set_title("Total Errores por Época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Errores")
    ax2.set_xticks(range(0, n_epochs, 2))
    ax2.grid(True)

    plt.tight_layout()

    # --- Guardar Frame en Memoria ---
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    frames.append(imageio.imread(buf))
    buf.close()

    plt.close()  # Cierra la figura para no acumular memoria

# --- 5. GUARDAR GIF ---
imageio.mimsave("gifs/entrega_primera_caso_1.gif", frames, fps=2)

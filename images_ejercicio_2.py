import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import scipy.ndimage as ndimage


def procesar_imagen_bordes_gordos():
    # ===========================================================
    # 1. OBTENCIÓN
    # ===========================================================
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    try:
        response = requests.get(url)
        response.raise_for_status()
        img_pil = Image.open(BytesIO(response.content))
        matriz_rgb = np.array(img_pil)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Escala de Grises
    r, g, b = matriz_rgb[:, :, 0], matriz_rgb[:, :, 1], matriz_rgb[:, :, 2]
    matriz_gris = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # A. Umbralizado
    UMBRAL = 100
    matriz_binaria_sucia = np.where(matriz_gris > UMBRAL, 255.0, 0.0)

    # B. Limpieza Morfológica (Apertura)
    # Tu configuración de 15x15 es agresiva pero efectiva para limpiar el fondo
    estructura_limpieza = np.ones((15, 15))
    matriz_limpia = ndimage.binary_opening(
        matriz_binaria_sucia, structure=estructura_limpieza)
    matriz_limpia = matriz_limpia.astype(float) * 255

    # C. Detección de Bordes (Laplaciano)
    kernel_laplaciano = np.array([
        [0, -1,  0],
        [-1, 4, -1],
        [0, -1,  0]
    ])

    bordes_raw = ndimage.convolve(matriz_limpia, kernel_laplaciano)
    bordes_relu = np.maximum(bordes_raw, 0)  # ReLU (quita negativos)

    # ===========================================================
    # D. ENGROSAMIENTO DEL BORDE (NUEVO PASO)
    # ===========================================================
    # Usamos dilatación binaria.
    # 'iterations' define cuántas veces expandimos la línea.
    # 1 iteración = un poco más gordo. 3 iteraciones = muy gordo (tipo rotulador).
    GROSOR = 2

    # Convertimos a booleano (>0) para dilatar y luego volvemos a número
    bordes_gordos = ndimage.binary_dilation(bordes_relu > 0, iterations=GROSOR)
    bordes_final = bordes_gordos.astype(float) * 255

    # ===========================================================
    # 2. VISUALIZACIÓN
    # ===========================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Foto 1: Original
    axes[0].imshow(matriz_rgb)
    axes[0].set_title("1. Original (RGB)")
    axes[0].axis('off')

    # Foto 2: Silueta
    axes[1].imshow(matriz_limpia, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f"2. Silueta Limpia")
    axes[1].axis('off')

    # Foto 3: Bordes Gordos
    axes[2].imshow(bordes_final, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f"3. Bordes Finales\n(Engrosado: {GROSOR} iteraciones)")
    axes[2].axis('off')

    plt.tight_layout()
    print("Mostrando resultado con bordes más gruesos...")
    plt.show()


if __name__ == "__main__":
    procesar_imagen_bordes_gordos()

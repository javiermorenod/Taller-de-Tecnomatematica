import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import scipy.ndimage as ndimage


def procesar_imagen_final_3_fotos():
    # ===========================================================
    # 1. PROCESAMIENTO (Igual que el paso anterior)
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

    # A. Umbralizado (Binarización inicial - aún ruidosa)
    UMBRAL = 100
    matriz_binaria_sucia = np.where(matriz_gris > UMBRAL, 255.0, 0.0)

    # B. Limpieza Morfológica (Apertura)
    estructura = np.ones((15, 15))
    matriz_limpia = ndimage.binary_opening(
        matriz_binaria_sucia, structure=estructura)
    matriz_limpia = matriz_limpia.astype(float) * 255

    # C. Detección de Bordes (Sobre la imagen limpia)
    kernel_laplaciano = np.array([
        [0, -1,  0],
        [-1, 4, -1],
        [0, -1,  0]
    ])

    bordes_raw = ndimage.convolve(matriz_limpia, kernel_laplaciano)
    bordes_relu = np.maximum(bordes_raw, 0)  # ReLU
    bordes_final = np.clip(bordes_relu, 0, 255)  # Clip final

    # ===========================================================
    # 2. VISUALIZACIÓN (Solo 3 fotos)
    # ===========================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Foto 1: Original
    axes[0].imshow(matriz_rgb)
    axes[0].set_title("1. Original (RGB)")
    axes[0].axis('off')

    # Foto 2: Intermedia Limpia (Resultado de la morfología)
    axes[1].imshow(matriz_limpia, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(
        f"2. Silueta Limpia\n(Umbral {UMBRAL} + Apertura Morfológica)")
    axes[1].axis('off')

    # Foto 3: Bordes Finales
    axes[2].imshow(bordes_final, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title("3. Bordes Finales (ReLU)")
    axes[2].axis('off')

    plt.tight_layout()
    print("Mostrando las 3 imágenes del proceso...")
    plt.show()


if __name__ == "__main__":
    procesar_imagen_final_3_fotos()

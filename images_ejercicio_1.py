import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO


def procesar_imagen():
    # -----------------------------------------------------------
    # 1. CARGA: Descargar imagen y convertir a Matriz RGB
    # -----------------------------------------------------------
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    print(f"Descargando imagen de: {url}...")

    try:
        response = requests.get(url)
        response.raise_for_status()

        # Convertimos los bytes descargados en un objeto imagen
        imagen_pil = Image.open(BytesIO(response.content))

        # Transformamos la imagen a una matriz NumPy (Alto x Ancho x 3 Canales)
        matriz_rgb = np.array(imagen_pil)
        print("Imagen descargada y convertida a matriz exitosamente.")

    except Exception as e:
        print(f"Error: {e}")
        return

    # -----------------------------------------------------------
    # 2. PROCESAMIENTO: Escala de Grises (Operación Matricial)
    # -----------------------------------------------------------
    # Fórmula de luminancia: Y = 0.299*R + 0.587*G + 0.114*B
    # Multiplicamos cada canal de color por su peso específico
    r, g, b = matriz_rgb[:, :, 0], matriz_rgb[:, :, 1], matriz_rgb[:, :, 2]
    matriz_gris = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # -----------------------------------------------------------
    # 3. PROCESAMIENTO: Invertir Colores (Negativo)
    # -----------------------------------------------------------
    # Restamos el valor actual al valor máximo (255)
    matriz_invertida = 255.0 - matriz_gris

    # -----------------------------------------------------------
    # 4. SALIDA: Visualización de resultados
    # -----------------------------------------------------------
    # Creamos una figura con 3 sub-gráficos
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # A) Imagen Original
    axes[0].imshow(matriz_rgb)
    axes[0].set_title("Original (RGB)")
    axes[0].axis('off')

    # B) Escala de Grises
    # cmap='gray' es necesario para interpretar la matriz 2D como grises
    axes[1].imshow(matriz_gris, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Escala de Grises")
    axes[1].axis('off')

    # C) Escala de Grises Invertida
    axes[2].imshow(matriz_invertida, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title("Grises Invertida (Negativo)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    procesar_imagen()

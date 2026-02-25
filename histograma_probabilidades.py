import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

def predict_handdrawn_png(
    model,
    image_path: str,
    device=None,
    invert: bool = False,
    show: bool = True,
):
    """
    Carga una imagen (png/jpg), la convierte a formato MNIST (1x28x28 en [0,1]),
    y devuelve la predicción (argmax) + vector de probabilidades (softmax).

    Parámetros
    ----------
    model : tu AEWithSoftmaxHead (o similar) ya entrenado
    image_path : ruta a la imagen
    device : torch.device (si None, intenta inferir)
    invert : si tu dibujo es negro sobre blanco normalmente NO hace falta.
    Si el modelo te predice mal, prueba invert=True.
    show : si True, muestra imagen procesada + histograma de probabilidades.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # 1) Cargar imagen y pasar a escala de grises
    img = Image.open(image_path).convert("L")

    # 2) (Opcional) invertir colores
    if invert:
        img = ImageOps.invert(img)

    # 3) Redimensionar a 28x28 (misma dimensión MNIST)
    img_28 = img.resize((28, 28), resample=Image.BILINEAR)

    # 4) Normalizar a [0,1] y dar forma (1,1,28,28)
    x = np.array(img_28, dtype=np.float32) / 255.0
    x_t = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,28,28]

    # 5) Forward: obtener y_hat
    with torch.no_grad():
        z, x_hat, logits, y_hat = model(x_t)

    probs = y_hat.squeeze(0).detach().cpu().numpy()  # [10]
    pred = int(np.argmax(probs))

    if show:
        # Imagen procesada (la que realmente "ve" el modelo)
        plt.figure()
        plt.imshow(x, cmap="gray")
        plt.title(f"Imagen procesada 28x28 | pred={pred}")
        plt.axis("off")
        plt.show()

        # Histograma de probabilidades
        plt.figure()
        plt.bar(range(len(probs)), probs)
        plt.xticks(range(len(probs)))
        plt.ylim(0, 1)
        plt.title("Probabilidades (softmax)")
        plt.xlabel("Dígito")
        plt.ylabel("P(y=dígito)")
        plt.grid(True, axis="y")
        plt.show()

    print(f"Predicción: {pred}")
    print("Probabilidades:", np.round(probs, 4))

    return pred, probs, x_t

# USO:
pred, probs, x_t = predict_handdrawn_png(model, "dos.png", invert=False)
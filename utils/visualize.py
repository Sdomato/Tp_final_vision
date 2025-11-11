import torch
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import random
import numpy as np

def visualize_colorization(model, dataloader, device=None, num_samples=3, figsize=(10,4)):
    """
    Muestra ejemplos de colorización (entrada, color real y color predicho).

    Parámetros
    ----------
    model : nn.Module
        Modelo ya entrenado (debe devolver canales a,b).
    dataloader : DataLoader
        DataLoader de validación o test.
    device : torch.device (opcional)
        Si no se pasa, se detecta automáticamente (CPU/GPU).
    num_samples : int
        Cantidad de ejemplos aleatorios a mostrar.
    figsize : tuple
        Tamaño de la figura matplotlib.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Obtener un batch de validación
    L_batch, ab_real_batch = next(iter(dataloader))
    L_batch, ab_real_batch = L_batch.to(device), ab_real_batch.to(device)

    with torch.no_grad():
        ab_pred_batch = model(L_batch).cpu()

    # Función auxiliar
    def lab_to_rgb(L, ab):
        lab = torch.cat((L, ab), dim=0).permute(1, 2, 0).numpy()
        return lab2rgb(lab * [100, 128, 128])

    # Mostrar ejemplos aleatorios
    indices = random.sample(range(len(L_batch)), min(num_samples, len(L_batch)))

    for i in indices:
        L_sample = L_batch[i].cpu()
        ab_real_sample = ab_real_batch[i].cpu()
        ab_pred_sample = ab_pred_batch[i]

        rgb_real = lab_to_rgb(L_sample, ab_real_sample)
        rgb_pred = lab_to_rgb(L_sample, ab_pred_sample)

        plt.figure(figsize=figsize)
        plt.subplot(1,3,1)
        plt.imshow(L_sample.squeeze(), cmap='gray')
        plt.title("Entrada (Grises)")
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(rgb_real)
        plt.title("Color real")
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(rgb_pred)
        plt.title("Color predicho")
        plt.axis('off')

        plt.show()



def visualize_ranked_colorizations(model, dataloader, device=None, num_each=2, criterion=None):
    """
    Muestra ejemplos de colorización ordenados por error (mejores, peores y medios).

    Parámetros
    ----------
    model : nn.Module
        Modelo entrenado.
    dataloader : DataLoader
        DataLoader de validación.
    device : torch.device
        CPU o GPU.
    num_each : int
        Número de ejemplos a mostrar por categoría (mejores, medios, peores).
    criterion : función de pérdida (opcional)
        Si no se pasa, usa L1 (MAE).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    criterion = criterion or torch.nn.L1Loss(reduction='none')  # pixel-wise loss

    all_samples = []

    with torch.no_grad():
        for L_batch, ab_real_batch in dataloader:
            L_batch, ab_real_batch = L_batch.to(device), ab_real_batch.to(device)
            ab_pred_batch = model(L_batch)

            # Calcular error por imagen (promedio del L1 pixelwise)
            losses = criterion(ab_pred_batch, ab_real_batch).mean(dim=[1,2,3]).cpu().numpy()
            
            for i in range(len(L_batch)):
                all_samples.append({
                    "L": L_batch[i].cpu(),
                    "ab_real": ab_real_batch[i].cpu(),
                    "ab_pred": ab_pred_batch[i].cpu(),
                    "loss": losses[i]
                })

    # Ordenar por error (menor = mejor)
    all_samples.sort(key=lambda x: x["loss"])
    total = len(all_samples)
    mid_start = total // 2 - num_each // 2

    selected = (
        all_samples[:num_each] +                        # mejores
        all_samples[mid_start:mid_start + num_each] +   # del medio
        all_samples[-num_each:]                         # peores
    )

    titles = (
        [f"Mejor {i+1}" for i in range(num_each)] +
        [f"Medio {i+1}" for i in range(num_each)] +
        [f"Peor {i+1}" for i in range(num_each)]
    )

    def lab_to_rgb(L, ab):
        lab = torch.cat((L, ab), dim=0).permute(1, 2, 0).numpy()
        return np.clip(lab2rgb(lab * [100, 128, 128]), 0, 1)

    # Mostrar imágenes
    ncols = 3
    nrows = len(selected)
    plt.figure(figsize=(ncols * 3, nrows * 3))

    for i, s in enumerate(selected):
        L, ab_real, ab_pred = s["L"], s["ab_real"], s["ab_pred"]
        rgb_real = lab_to_rgb(L, ab_real)
        rgb_pred = lab_to_rgb(L, ab_pred)

        # Entrada gris
        plt.subplot(nrows, ncols, i * 3 + 1)
        plt.imshow(L.squeeze(), cmap="gray")
        plt.title(f"{titles[i]} - Entrada")
        plt.axis('off')

        # Real
        plt.subplot(nrows, ncols, i * 3 + 2)
        plt.imshow(rgb_real)
        plt.title("Real")
        plt.axis('off')

        # Predicho
        plt.subplot(nrows, ncols, i * 3 + 3)
        plt.imshow(rgb_pred)
        plt.title(f"Predicho\nLoss={s['loss']:.4f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

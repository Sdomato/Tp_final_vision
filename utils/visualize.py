import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.color import lab2rgb

# ===============================================
#   Visualización simple de colorización
# ===============================================
def visualize_colorization(model, dataloader, device=None, num_samples=3, figsize=(10,4)):
    """
    Muestra ejemplos de colorización (entrada, color real y color predicho).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Obtener un batch de validación
    L_batch, ab_real_batch = next(iter(dataloader))
    L_batch, ab_real_batch = L_batch.to(device), ab_real_batch.to(device)

    with torch.no_grad():
        ab_pred_batch = model(L_batch).cpu()

    def lab_to_rgb(L, ab):
        lab = torch.cat((L, ab), dim=0).permute(1, 2, 0).numpy()
        return np.clip(lab2rgb(lab * [100, 128, 128]), 0, 1)

    indices = random.sample(range(len(L_batch)), min(num_samples, len(L_batch)))

    fig, axes = plt.subplots(len(indices), 3, figsize=(figsize[0], figsize[1] * len(indices)))
    if len(indices) == 1:
        axes = [axes]  # mantener estructura consistente

    for row, i in enumerate(indices):
        L_sample = L_batch[i].cpu()
        ab_real_sample = ab_real_batch[i].cpu()
        ab_pred_sample = ab_pred_batch[i]

        rgb_real = lab_to_rgb(L_sample, ab_real_sample)
        rgb_pred = lab_to_rgb(L_sample, ab_pred_sample)

        axes[row][0].imshow(L_sample.squeeze(), cmap='gray')
        axes[row][0].set_title("Entrada (Grises)")
        axes[row][0].axis('off')

        axes[row][1].imshow(rgb_real)
        axes[row][1].set_title("Color real")
        axes[row][1].axis('off')

        axes[row][2].imshow(rgb_pred)
        axes[row][2].set_title("Color predicho")
        axes[row][2].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close('all')
    torch.cuda.empty_cache()


def visualize_ranked_colorizations(model, dataloader, device=None, num_each=2, criterion=None):
    """
    Muestra ejemplos de colorización ordenados por error (mejores, medios y peores).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    criterion = criterion or torch.nn.L1Loss(reduction='none')

    all_samples = []

    with torch.no_grad():
        for L_batch, ab_real_batch in dataloader:
            L_batch, ab_real_batch = L_batch.to(device), ab_real_batch.to(device)
            ab_pred_batch = model(L_batch)

            # Calcular error por imagen
            losses = criterion(ab_pred_batch, ab_real_batch).mean(dim=[1,2,3]).cpu().numpy()

            for i in range(len(L_batch)):
                all_samples.append({
                    "L": L_batch[i].cpu(),
                    "ab_real": ab_real_batch[i].cpu(),
                    "ab_pred": ab_pred_batch[i].cpu(),
                    "loss": float(losses[i])
                })

    # Ordenar
    all_samples.sort(key=lambda x: x["loss"])
    total = len(all_samples)
    mid_start = total // 2 - num_each // 2

    selected = (
        all_samples[:num_each] +                        # mejores
        all_samples[mid_start:mid_start + num_each] +   # medios
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

    # Mostrar todas en una sola figura
    ncols = 3
    nrows = len(selected)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, s in enumerate(selected):
        L, ab_real, ab_pred = s["L"], s["ab_real"], s["ab_pred"]
        rgb_real = lab_to_rgb(L, ab_real)
        rgb_pred = lab_to_rgb(L, ab_pred)

        axes[i, 0].imshow(L.squeeze(), cmap="gray")
        axes[i, 0].set_title(f"{titles[i]} - Entrada")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(rgb_real)
        axes[i, 1].set_title("Real")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(rgb_pred)
        axes[i, 2].set_title(f"Predicho\nLoss={s['loss']:.4f}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close('all')
    torch.cuda.empty_cache()

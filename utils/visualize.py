import torch
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import random

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

import torch
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Función para Graficar (La que te pasé antes) ---
def plot_ab_distribution(real_ab, pred_ab_viejo, pred_ab_nuevo):
    # Tomamos 2000 píxeles al azar para que el gráfico sea rápido y legible
    if len(real_ab.flatten()) > 0:
        idx = np.random.choice(real_ab.shape[0], min(2000, real_ab.shape[0]), replace=False)
    else:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    # Configuración de ejes (ajusta si tus datos son de 0 a 1)
    # Asumimos rango [-1, 1] por el Tanh
    limit = 1.1 
    
    datasets = [
        (real_ab, 'green', 'Ground Truth (Realidad)'),
        (pred_ab_viejo, 'red', 'Modelo Base (UNet Simple)'),
        (pred_ab_nuevo, 'blue', 'Modelo Final (ResNet34 + Stats)')
    ]

    for i, (data, color, title) in enumerate(datasets):
        # Canal A en eje X, Canal B en eje Y
        a_vals = data[idx, 0]
        b_vals = data[idx, 1]
        
        axes[i].scatter(a_vals, b_vals, alpha=0.3, c=color, s=5)
        axes[i].set_title(title)
        axes[i].set_xlim(-limit, limit)
        axes[i].set_ylim(-limit, limit)
        axes[i].set_xlabel("Canal A (Verde-Rojo)")
        if i == 0: axes[i].set_ylabel("Canal B (Azul-Amarillo)")
        axes[i].axhline(0, color='black', lw=0.5, ls='--')
        axes[i].axvline(0, color='black', lw=0.5, ls='--')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- 2. Función para Extraer los Datos ---
def get_data_for_plot(loader, model_viejo, model_nuevo, device):
    model_viejo.eval()
    model_nuevo.eval()
    
    # Sacamos UN SOLO batch del val_loader (suficiente para el gráfico)
    L, ab_real = next(iter(loader))
    L = L.to(device)
    
    with torch.no_grad():
        # Predicción Modelo Viejo
        pred_viejo = model_viejo(L)
        
        # Predicción Modelo Nuevo
        pred_nuevo = model_nuevo(L)

    # Convertimos a Numpy y ordenamos los canales
    # Salida shape: (N_pixeles, 2)
    # .permute(0, 2, 3, 1) cambia de (B, C, H, W) a (B, H, W, C)
    real_numpy = ab_real.permute(0, 2, 3, 1).reshape(-1, 2).cpu().numpy()
    viejo_numpy = pred_viejo.permute(0, 2, 3, 1).reshape(-1, 2).cpu().numpy()
    nuevo_numpy = pred_nuevo.permute(0, 2, 3, 1).reshape(-1, 2).cpu().numpy()
    
    return real_numpy, viejo_numpy, nuevo_numpy
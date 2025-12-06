import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_ab_distribution(real_ab, pred_ab_viejo, pred_ab_nuevo, pred_ab_hist):
    
    if len(real_ab.flatten()) > 0:
        idx = np.random.choice(real_ab.shape[0], min(2000, real_ab.shape[0]), replace=False)
    else:
        return

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
    
    limit = 1.1 
    
    datasets = [
        (real_ab, 'green', 'Ground Truth (Realidad)'),
        (pred_ab_viejo, 'red', 'Baseline (UNet Simple)'),
        (pred_ab_nuevo, 'blue', 'ResNet34 Frozen'),
        (pred_ab_hist, 'purple', 'ResNet34 + Histograma')
    ]

    for i, (data, color, title) in enumerate(datasets):
        a_vals = data[idx, 0]
        b_vals = data[idx, 1]
        
        axes[i].scatter(a_vals, b_vals, alpha=0.3, c=color, s=5)
        axes[i].set_title(title, fontsize=11)
        axes[i].set_xlim(-limit, limit)
        axes[i].set_ylim(-limit, limit)
        axes[i].set_xlabel("Canal A (Verde-Rojo)")
        
        if i == 0: 
            axes[i].set_ylabel("Canal B (Azul-Amarillo)")
            
        axes[i].axhline(0, color='black', lw=0.5, ls='--')
        axes[i].axvline(0, color='black', lw=0.5, ls='--')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_saturation_hist(loader, models_list, labels_list, device):
    
    sat_real = []
    sat_preds = [[] for _ in models_list] 

    for m in models_list:
        m.eval()


    with torch.no_grad():
        for i, (L, ab_real) in enumerate(loader):
            if i >= 10: break 
            
            L = L.to(device)
            ab_real = ab_real.to(device)

            sat_r = torch.sqrt(ab_real[:, 0]**2 + ab_real[:, 1]**2)
            sat_real.extend(sat_r.flatten().cpu().numpy())

            for idx, model in enumerate(models_list):
                pred = model(L)
                sat_p = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2)
                sat_preds[idx].extend(sat_p.flatten().cpu().numpy())

    plt.figure(figsize=(10, 6))
    bins = 60
    alpha = 0.3

    plt.hist(sat_real, bins=bins, alpha=alpha, color='green', label='Realidad (GT)', density=True, histtype='stepfilled')
    plt.hist(sat_real, bins=bins, alpha=1.0, color='green', density=True, histtype='step', linewidth=1.5) # Borde

    colors = ['red', 'blue', 'purple', 'orange']

    for i, sat_data in enumerate(sat_preds):
        color = colors[i % len(colors)] 
        label = labels_list[i]
        
        plt.hist(sat_data, bins=bins, alpha=alpha, color=color, label=label, density=True, histtype='stepfilled')
        plt.hist(sat_data, bins=bins, alpha=1.0, color=color, density=True, histtype='step', linewidth=1.5) # Borde

    plt.title("Comparación de Viveza del Color (Saturación)", fontsize=14)
    plt.xlabel("Nivel de Saturación (0 = Gris, 1 = Color Puro)", fontsize=12)
    plt.ylabel("Densidad de Píxeles", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.0)
    plt.show()

def get_data_for_plot(loader, model_viejo, model_nuevo, model_hist, device):
    model_viejo.eval()
    model_nuevo.eval()
    model_hist.eval()
    
    L, ab_real = next(iter(loader))
    L = L.to(device)
    
    with torch.no_grad():
    
        pred_viejo = model_viejo(L)
        pred_nuevo = model_nuevo(L)
        pred_hist = model_hist(L)

    real_numpy = ab_real.permute(0, 2, 3, 1).reshape(-1, 2).cpu().numpy()
    viejo_numpy = pred_viejo.permute(0, 2, 3, 1).reshape(-1, 2).cpu().numpy()
    nuevo_numpy = pred_nuevo.permute(0, 2, 3, 1).reshape(-1, 2).cpu().numpy()
    hist_numpy = pred_hist.permute(0, 2, 3, 1).reshape(-1, 2).cpu().numpy()
    
    return real_numpy, viejo_numpy, nuevo_numpy, hist_numpy
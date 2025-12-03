# utils_hist.py
import torch
import torch.nn.functional as F
import numpy as np
from skimage import color
import matplotlib.pyplot as plt 

# ============================================================
# Histograma en espacio [-1,1]
# ============================================================

K = 32  # número de bins

bin_centers = torch.linspace(-1.0, 1.0, steps=K)  # 32 bins en [-1,1]

def ab_to_bins(ab):
    """
    ab: tensor (B,2,H,W) normalizado en [-1,1]
    Devuelve:
      idx_a, idx_b  (B, H, W) long
    """
    a = ab[:,0,:,:]
    b = ab[:,1,:,:]

    a_norm = (a + 1) / 2  # pasa [-1,1] -> [0,1]
    b_norm = (b + 1) / 2

    idx_a = torch.clamp((a_norm * (K-1)).long(), 0, K-1)
    idx_b = torch.clamp((b_norm * (K-1)).long(), 0, K-1)

    return idx_a, idx_b


def hist_loss(logits_a, logits_b, idx_a, idx_b):
    return F.cross_entropy(logits_a, idx_a) + F.cross_entropy(logits_b, idx_b)


def logits_to_ab(logits_a, logits_b):
    """
    logits_(B,K,H,W) → ab (B,2,H,W) en [-1,1]
    """
    device = logits_a.device
    probs_a = F.softmax(logits_a, dim=1)
    probs_b = F.softmax(logits_b, dim=1)

    centers = bin_centers.to(device).view(1, K, 1, 1)

    a_exp = torch.sum(probs_a * centers, dim=1)
    b_exp = torch.sum(probs_b * centers, dim=1)

    return torch.stack([a_exp, b_exp], dim=1)  # (B,2,H,W)


# ============================================================
# Conversión Lab (usando normas de TU dataset)
# ============================================================

def lab_to_rgb_from_norm(L, ab):
    """
    L: (B,1,H,W) en [0,1]
    ab: (B,2,H,W) en [-1,1]
    Devuelve: (B,H,W,3) en [0,1]
    """
    L_np = (L.squeeze(1).cpu().numpy() * 100.0)
    ab_np = ab.cpu().numpy() * 128.0   # desnormalizar [-1,1] → [-128,128]

    B = L_np.shape[0]
    rgbs = []

    for i in range(B):
        lab_img = np.stack([L_np[i], ab_np[i,0], ab_np[i,1]], axis=-1)
        rgb = color.lab2rgb(lab_img)
        rgbs.append(rgb)

    return np.stack(rgbs, axis=0)




def show_colorization(model, dataloader, device="cuda"):
    model.eval()
    L, ab = next(iter(dataloader))  # toma un batch
    L, ab = L.to(device), ab.to(device)

    with torch.no_grad():
        logits_a, logits_b = model(L)
        pred_ab = logits_to_ab(logits_a, logits_b)

    # convertir a RGB (numpy)
    pred_rgb = lab_to_rgb_from_norm(L.cpu(), pred_ab.cpu())  # (B,H,W,3)
    gt_rgb   = lab_to_rgb_from_norm(L.cpu(), ab.cpu())       # (B,H,W,3)
    L_np = L.squeeze(1).cpu().numpy()  # (B,H,W)

    # mostrar primeros 4
    n = min(4, L_np.shape[0])
    plt.figure(figsize=(12, 9))

    for i in range(n):
        # Grayscale
        plt.subplot(n, 3, 3*i+1)
        plt.imshow(L_np[i], cmap="gray")
        plt.title("L (gris)")
        plt.axis("off")

        # Predicho
        plt.subplot(n, 3, 3*i+2)
        plt.imshow(pred_rgb[i])
        plt.title("Predicción")
        plt.axis("off")

        # Truth
        plt.subplot(n, 3, 3*i+3)
        plt.imshow(gt_rgb[i])
        plt.title("Ground Truth")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

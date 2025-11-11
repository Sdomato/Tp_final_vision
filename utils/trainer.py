import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from pytorch_msssim import ssim
from lpips import LPIPS

def trainer(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=1e-3,
    criterion="l1",   # puede ser "l1", "ssim", "combined" o función personalizada
    device=None,
    save_path="checkpoints",
    save_name="model_color.pt"
):
    """
    Entrena un modelo de colorización con soporte para distintos criterios.
    """

    # --- Configuración de dispositivo ---
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- Función auxiliar para obtener el criterio ---
    def get_criterion(name):
        if callable(name):  # si ya es función
            return name

        if isinstance(name, str):
            name = name.lower()
        else:
            raise TypeError(f"❌ Tipo de criterio inválido: {type(name)}. Usa 'l1', 'ssim', 'combined' o una función.")

        if name == "l1":
            return nn.L1Loss()

        elif name == "ssim":
            if ssim is None:
                raise ImportError("Para usar SSIM necesitás instalar pytorch-msssim (`pip install pytorch-msssim`).")
            def ssim_loss(pred, target):
                return 1 - ssim(pred, target, data_range=1.0, size_average=True)
            return ssim_loss

        elif name == "combined":
            if ssim is None:
                raise ImportError("Para usar Combined necesitás pytorch-msssim (`pip install pytorch-msssim`).")
            l1 = nn.L1Loss()
            def combined_loss(pred, target):
                return 0.85 * l1(pred, target) + 0.15 * (1 - ssim(pred, target, data_range=1.0, size_average=True))
            return combined_loss
        
        elif name == "perceptual":
            if LPIPS is None:
                raise ImportError("Para usar LPIPS necesitás instalar lpips (`pip install lpips`).")
            l1 = nn.L1Loss()
            lpips_fn = LPIPS(net='alex').to(device)
            lpips_fn.freeze() 

            def perceptual_loss(pred, target):
                pred_norm = pred * 2.0 - 1.0
                target_norm = target * 2.0 - 1.0
                l1_val = l1(pred, target)
                lpips_val = lpips_fn(pred_norm, target_norm).mean() 
                return 0.85 * l1_val + 0.15 * lpips_val
            
            return perceptual_loss


        else:
            raise ValueError(f"❌ Criterio '{name}' no reconocido. Usa 'l1', 'ssim', 'combined' o una función.")

    # Obtener la función de pérdida real
    criterion_fn = get_criterion(criterion)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    print(f"Entrenando en: {device}")
    print(f"Usando criterio: {criterion if isinstance(criterion, str) else 'custom function'}")
    print("=" * 50)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # --- Loop de entrenamiento ---
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Época {epoch}/{epochs}")

        for L, ab in pbar:
            L, ab = L.to(device, non_blocking=True), ab.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(L)
                loss = criterion_fn(out, ab)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)

        # --- Validación ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for L, ab in val_loader:
                L, ab = L.to(device, non_blocking=True), ab.to(device, non_blocking=True)
                out = model(L)
                val_loss += criterion_fn(out, ab).item()

        val_loss /= len(val_loader)
        print(f"Época {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # --- Guardar mejor modelo ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Path(save_path) / save_name)
            print(f"✅ Nuevo mejor modelo guardado (Val Loss={val_loss:.4f})")

    print("=" * 50)
    print("Entrenamiento finalizado.")
    print(f"Mejor Val Loss: {best_val_loss:.4f}")

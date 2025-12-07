import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from pytorch_msssim import ssim
from utils.histogram_loss import ColorStatsLoss

def get_criterion_by_name(name, device):
    """Factory para obtener la función de pérdida."""
    if callable(name): return name
    
    name = name.lower()
    
    if name == "l1":
        return nn.L1Loss()
    
    elif name == "mse":
        return nn.MSELoss()

    elif name == "ssim":
        def ssim_loss(pred, target):
            return 1 - ssim(pred, target, data_range=1.0, size_average=True)
        return ssim_loss

    elif name == "combined":
        # L1 + SSIM
        l1 = nn.L1Loss()
        def combined_loss(pred, target):
            return 0.85 * l1(pred, target) + 0.15 * (1 - ssim(pred, target, data_range=1.0, size_average=True))
        return combined_loss
    
    elif name == "histogram":
        if ColorStatsLoss is None: raise ImportError("ColorStatsLoss no definida.")
        l1_loss = nn.L1Loss()
        stats_loss = ColorStatsLoss().to(device)
        def hist_loss_fn(pred, target):
            return 0.8 * l1_loss(pred, target) + 0.2 * stats_loss(pred, target)
        return hist_loss_fn
    else:
        raise ValueError(f"❌ Criterio '{name}' no reconocido.")


def trainer(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=2e-4,
    criterion="l1",
    device=None,
    save_path="checkpoints",
    save_name="model_color.pt"
):
    # --- Configuración ---
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    #  función de pérdida
    criterion_fn = get_criterion_by_name(criterion, device)
    
    # Optimizador
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Directorios
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Métricas
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "best_val_loss": None}
    
    print(f"🚀 Iniciando entrenamiento en: {device}")
    print(f"📉 Criterio: {criterion}")
    print("=" * 60)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{epochs} [Train]")

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

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for L, ab in val_loader:
                L, ab = L.to(device, non_blocking=True), ab.to(device, non_blocking=True)
                out = model(L)
                
                val_loss += criterion_fn(out, ab).item()

        val_loss /= len(val_loader)
        
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_val_loss"] = best_val_loss
            torch.save(model.state_dict(), Path(save_path) / save_name)

    print("=" * 60)
    print(f"🏁 Entrenamiento finalizado. Mejor Val Loss: {best_val_loss:.4f}")
    return history


def train_model(model, train_loader, val_loader, save_name, criterion="l1", force_train=False, epochs=10):
    """
    Wrapper inteligente que evita reentrenar si ya existe el modelo,
    a menos que force_train=True.
    """
    save_path = Path("pesos_entrenados")
    history_path = Path("loss_vs_epoch")
    
    model_file = save_path / save_name
    history_file = history_path / f"{save_name}_history.pt"

    save_path.mkdir(exist_ok=True)
    history_path.mkdir(exist_ok=True)

    if model_file.exists() and not force_train:
        print(f"✅ Modelo encontrado en '{model_file}'.")
        print("⏭️ Saltando entrenamiento (usa force_train=True para reentrenar).")
        
        if history_file.exists():
            print("📈 Historial cargado.")
            return torch.load(history_file, map_location="cpu")
        else:
            print("⚠️ Historial no encontrado.")
            return None

    else:
        if force_train:
            print(f"🔄 Forzando re-entrenamiento de '{save_name}'...")
        else:
            print(f"🆕 Modelo no encontrado. Entrenando '{save_name}'...")

        history = trainer(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            save_path=save_path,
            save_name=save_name,
            criterion=criterion,
        )
        torch.save(history, history_file)
        print(f"📈 Historial guardado en '{history_file}'")

        return history
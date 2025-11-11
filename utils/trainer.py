import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from pathlib import Path

def trainer(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=1e-3,
    criterion=None,
    device=None,
    save_path="checkpoints",
    save_name="model_color.pt"
):
    """
    Entrena un modelo de colorización dado un dataloader de entrenamiento y validación.

    Parámetros:
    ------------
    model : nn.Module
        Modelo a entrenar (por ej. FastColorNet o UNetColor)
    train_loader : DataLoader
        Dataloader con datos de entrenamiento (L -> ab)
    val_loader : DataLoader
        Dataloader con datos de validación
    epochs : int
        Número de épocas de entrenamiento
    lr : float
        Tasa de aprendizaje
    criterion : función de pérdida (por defecto usa L1Loss)
    device : torch.device (opcional)
        Si no se pasa, detecta automáticamente CPU o GPU
    save_path : str
        Carpeta donde guardar el modelo entrenado
    save_name : str
        Nombre del archivo del modelo
    """

    # --- Configuración de dispositivo ---
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- Criterio y optimizador ---
    criterion = criterion or nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    Path(save_path).mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    print(f"Entrenando en: {device}")
    print(f"Usando pérdida: {criterion.__class__.__name__}")
    print("=" * 50)

    # --- Loop de entrenamiento ---
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Época {epoch}/{epochs}")
        for L, ab in pbar:
            L, ab = L.to(device), ab.to(device)

            optimizer.zero_grad()
            out = model(L)
            loss = criterion(out, ab)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)

        # --- Validación ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for L, ab in val_loader:
                L, ab = L.to(device), ab.to(device)
                out = model(L)
                val_loss += criterion(out, ab).item()

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

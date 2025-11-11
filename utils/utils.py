from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
import matplotlib.pyplot as plt

class ImagewoofColorizationDataset(Dataset):
    def __init__(self, root_dir, img_size=224, split="train"):
        self.root_dir = Path(root_dir) / split

        # Buscar imágenes con varias extensiones posibles
        self.img_paths = []
        self.img_paths += list(self.root_dir.rglob("*.JPEG"))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        if len(self.img_paths) == 0:
            print(f"⚠️ No se encontraron imágenes en {self.root_dir}. Verificá la ruta y extensión de archivos.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = np.array(self.transform(img)).transpose(1, 2, 0)  # HWC

        # Convertir RGB → Lab
        lab = rgb2lab(img).astype("float32")
        L = lab[:, :, 0] / 100.0          # Normalizar 0–1
        ab = lab[:, :, 1:] / 128.0        # Normalizar -1–1 aprox.

        # Convertir a tensores
        L = torch.from_numpy(L).unsqueeze(0)         # (1, H, W)
        ab = torch.from_numpy(ab).permute(2, 0, 1)   # (2, H, W)
        return L, ab

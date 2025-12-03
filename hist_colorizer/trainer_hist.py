# trainer_hist.py
import torch
from tqdm import tqdm
import numpy as np
from hist_colorizer.utils_hist import ab_to_bins, hist_loss, logits_to_ab, lab_to_rgb_from_norm

class TrainerHist:
    def __init__(self, model, lr=1e-4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for L, ab in tqdm(loader, desc="Train"):
            L = L.to(self.device)
            ab = ab.to(self.device)

            logits_a, logits_b = self.model(L)

            idx_a, idx_b = ab_to_bins(ab)
            idx_a, idx_b = idx_a.to(self.device), idx_b.to(self.device)

            loss = hist_loss(logits_a, logits_b, idx_a, idx_b)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        psnrs = []

        with torch.no_grad():
            for L, ab in tqdm(loader, desc="Val"):
                L = L.to(self.device)
                ab = ab.to(self.device)

                logits_a, logits_b = self.model(L)
                idx_a, idx_b = ab_to_bins(ab)
                loss = hist_loss(logits_a, logits_b,
                                 idx_a.to(self.device), idx_b.to(self.device))
                total_loss += loss.item()

                pred_ab = logits_to_ab(logits_a, logits_b)
                pred_rgb = lab_to_rgb_from_norm(L.cpu(), pred_ab.cpu())

                # ground truth
                gt_rgb = lab_to_rgb_from_norm(L.cpu(), ab.cpu())

                mse = np.mean((pred_rgb - gt_rgb)**2)
                psnr = 10 * np.log10(1.0 / mse)
                psnrs.append(psnr)

        return total_loss / len(loader), np.mean(psnrs)

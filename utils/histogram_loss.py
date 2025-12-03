import torch
import torch.nn as nn

class ColorStatsLoss(nn.Module):
    """
    Compara las estadísticas (Media y Desviación Estándar) de los canales a*b*.
    Esto fuerza al modelo a tener una distribución de colores (histograma) similar al real,
    evitando el problema de los colores "lavados" o "sepia".
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):

        mu_pred = torch.mean(pred, dim=[2, 3]) 
        mu_target = torch.mean(target, dim=[2, 3])
    
        std_pred = torch.std(pred, dim=[2, 3])
        std_target = torch.std(target, dim=[2, 3])
        
        loss_mu = self.l1(mu_pred, mu_target)
        loss_std = self.l1(std_pred, std_target)
        
        return loss_mu + loss_std
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList([
            vgg[:4].eval(),
            vgg[4:9].eval(),
            vgg[9:16].eval(),
            vgg[16:23].eval()
        ])
        for bl in self.blocks:
            for p in bl.parameters():
                p.requires_grad = False
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss

class FinalColorizationLoss(nn.Module):
    def __init__(self, l1_weight=1.0, perc_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = VGGPerceptualLoss()
        self.l1_weight = l1_weight
        self.perc_weight = perc_weight

    def forward(self, pred_ab, real_ab, L_input):
        loss_pix = self.l1(pred_ab, real_ab)
        
        pred_3c = torch.cat([L_input, pred_ab], dim=1)
        real_3c = torch.cat([L_input, real_ab], dim=1)
        
        loss_perc = self.perceptual(pred_3c, real_3c)
        
        return (self.l1_weight * loss_pix) + (self.perc_weight * loss_perc)
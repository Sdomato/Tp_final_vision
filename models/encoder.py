import torch
import torch.nn as nn

class FastColorNet(nn.Module):
    def __init__(self):
        super(FastColorNet, self).__init__()

        # --- Encoder liviano ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # -> 32×64×64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # -> 64×32×32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),# -> 128×16×16
            nn.ReLU(),
        )

        # --- Decoder liviano ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

import torch
import torch.nn as nn

class UNetColor(nn.Module):
    def __init__(self):
        super(UNetColor, self).__init__()

        # --- Encoder ---
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # --- Decoder ---
        self.up1 = self.up_block(256, 128)
        self.up2 = self.up_block(256, 64)
        self.up3 = self.up_block(128, 32)
        self.out = nn.Conv2d(64, 2, 3, padding=1)
        self.tanh = nn.Tanh()

        self.pool = nn.MaxPool2d(2, 2)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder con skip connections
        d1 = self.up1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e1], dim=1)

        out = self.tanh(self.out(d3))
        return out

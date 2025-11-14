import segmentation_models_pytorch as smp
import torch.nn as nn


class UNetResNet34(nn.Module):
    """
    U-Net con backbone ResNet34 de segmentation_models_pytorch.

    Entrada:  (B, 1, H, W)  -> canal L normalizado
    Salida:   (B, 2, H, W)  -> canales a*b* normalizados en ~[-1, 1]
    """
    def __init__(self, pre_entrenado: bool = True, congelar_encoder: bool = False):
        super().__init__()

        # Modelo base U-Net + ResNet34
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pre_entrenado else None,
            in_channels=1,  # canal L
            classes=2       # canales a,b
        )

        # Opcional: congelar encoder para que sólo se entrenen los decoders
        if congelar_encoder:
            for p in self.unet.encoder.parameters():
                p.requires_grad = False

        # Activación final para mantener coherencia con el resto de modelos
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.unet(x)     # (B,2,H,W)
        x = self.act(x)      # limitar a [-1,1] aprox.
        return x


def get_model_unet_resnet34(pre_entrenado: bool = True, congelar_encoder: bool = False):
    """
    Helper para crear.
    """
    return UNetResNet34(pre_entrenado=pre_entrenado, congelar_encoder=congelar_encoder)

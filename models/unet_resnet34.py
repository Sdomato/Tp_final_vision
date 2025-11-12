import segmentation_models_pytorch as smp
import torch.nn as nn

def get_model_unet_resnet34(pre_entrenado=True):
    """
    Crea el modelo U-Net con un backbone ResNet34.
    """
    modelo_unet = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights="imagenet" if pre_entrenado else None, 
        in_channels=1,                  # (Canal L)
        classes=2                       # (Canales a*b*)
    )
    
    modelo_final = nn.Sequential(
        modelo_unet,
        nn.Tanh()  
    )
    
    return modelo_final
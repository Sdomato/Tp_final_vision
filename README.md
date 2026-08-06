# Colorización de imágenes con deep learning · TP Final de Visión Artificial

Colorización automática de imágenes en escala de grises: dada una imagen en blanco y negro, la red predice los canales de color. Se comparan varias arquitecturas y funciones de pérdida.

> Trabajo práctico final de **Visión Artificial** (Universidad de San Andrés).

## Modelos comparados

- **FastColorNet** — red convolucional liviana como baseline.
- **U-Net** — encoder-decoder con skip connections (`unetColor`).
- **U-Net + ResNet34** — U-Net con encoder ResNet34 preentrenado, en varias configuraciones:
  - pérdida **L1**
  - pérdida **SSIM** (estructural)
  - **combinada** (L1 + estructural)
  - encoder **congelado** (frozen) vs. fine-tuning
  - variante con corrección de **histograma**

Las curvas de entrenamiento (loss vs. época) de cada variante están en `loss_vs_epoch/`.

## Estructura

| Carpeta / archivo | Contenido |
|---|---|
| `final.ipynb` | Notebook principal: entrenamiento y evaluación |
| `models/` | Arquitecturas (`unet.py`, `unet_resnet34.py`, `encoder.py`) |
| `hist_colorizer/` | Colorización con corrección de histograma |
| `loss_vs_epoch/` | Historiales de entrenamiento de cada modelo (`.pt`) |
| `graficos/` | Scripts de graficado de resultados |
| `Inputs_propios/` | Imágenes propias usadas como prueba |

## Cómo correr

Abrir `final.ipynb`. El entrenamiento usa el dataset **Imagewoof**; las imágenes de resultado se generan al ejecutar el notebook. Dependencias principales: `torch`, `torchvision`, `numpy` y `matplotlib`.

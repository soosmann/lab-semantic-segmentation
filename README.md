# Semantic Segmentation with PyTorch

A deep learning lab exploring **pixel-level image classification** on **PASCAL VOC**. The project trains a simplified PSPNet model, studies class imbalance, and compares the predictions and inference times of pretrained segmentation networks.

## Experiments

- **PSPNet Lite:** A ResNet-18 backbone with dilated convolutions and a simplified pooling module, trained on 224 × 224 images across 21 classes, including background.
- **Class imbalance:** Compare standard and class-weighted cross-entropy loss, alongside experiments with data augmentation and minority-class sampling.
- **Evaluation:** Track loss, mean Intersection over Union (mIoU), per-class IoU, and visual segmentation results.
- **Pretrained models:** Compare FCN and DeepLabv3 with ResNet-50 and ResNet-101 backbones, exploring image transformations and CPU/GPU inference time.

## Repository Contents

- [Notebook](src/imagen-lab-3-Marcel-and-Carlos.ipynb): implementation, experiments, and visualizations.
- [Lab report](docs/DLISP_Lab3_Hofmann_Marcel.pdf): analysis and discussion of the results.

## Getting Started

Open the notebook in Jupyter, Google Colab, or Kaggle with a CUDA-capable GPU enabled. For a local environment, install:

```bash
pip install torch torchvision numpy scipy matplotlib pillow wget jupyter
```

Run the dataset download and extraction cell, then set `args.train_root` to the extracted `VOC` directory. The notebook currently uses `/kaggle/working/VOC` and `device = 'cuda'`. Review the `configuration` class to adjust training settings, then run the experiment cells in order.

## Acknowledgments

Based on the semantic segmentation lab by Marcos Escudero-Viñolo, with material adapted from Diego Ortego, as credited in the notebook.

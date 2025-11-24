# UNET-Segmentation

## Overview

This project focuses on segmenting brain structures from magnetic resonance(MR) images using a UNet-based deep learning model. The work is carried out on the preprocessed OASIS dataset. The goal is to accurately separate different anatomical regions in the brain from each scan. The segmentation accuracy of model will need to be validated and achieve > 0.9 DSC for all
labels.

## Dataset

This project uses the preprocessed OASIS MRI dataset, which provides 3D brain scans along with their corresponding segmentation masks. Each mask contains four labels, representing different anatomical regions in the brain.

| ![Image 1](figures/train_val_progress_loss.png) | ![Image 2](figures/train_val_progress_accuracy.png) | ![Image 3](figures/train_val_progress_auroc.png) |
| ----------------------------------------------- | --------------------------------------------------- | ------------------------------------------------ |

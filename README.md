# UNET-Segmentation

## Overview

This project focuses on segmenting brain structures from magnetic resonance(MR) images using a UNet-based deep learning model. The work is carried out on the preprocessed OASIS dataset. The goal is to accurately separate different anatomical regions in the brain from each scan. The segmentation accuracy of model will need to be validated and achieve > 0.9 DSC for all
labels.

## Dataset

This project uses the preprocessed OASIS MRI dataset, which provides 3D brain scans along with their corresponding segmentation masks. Each mask contains four labels, representing different anatomical regions in the brain.

<p align="center">
  <img src="./figures/case_441_slice_0.nii.png" width="45%">
  <br>
  <b>Original MRI Slice</b>
</p>

<p align="center">
  <img src="./figures/seg_441_slice_0.nii.png" width="45%">
  <br>
  <b>Ground-Truth Segmentation Mask</b>
</p>

## Result

### Test Set Dice Scores

| Class    | Dice Score |
| -------- | ---------- |
| 0        | 0.9956     |
| 1        | 0.8983     |
| 2        | 0.9375     |
| 3        | 0.9661     |
| **Mean** | **0.9494** |

### Image segmentation and ground-truth

<p align="center">
  <img src="/figures/download.png">
  <br>
  <b>Image segmentation and ground-truth</b>
</p>

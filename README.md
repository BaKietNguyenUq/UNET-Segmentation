# UNET-Segmentation

## Introduction

This project focuses on segmenting brain structures from magnetic resonance(MR) images using a UNet-based deep learning model. The work is carried out on the preprocessed OASIS dataset. The goal is to accurately separate different anatomical regions in the brain from each scan. The segmentation accuracy of model will need to be validated and achieve > 0.9 DSC for all
labels.

## Dataset Overview

This project uses the preprocessed OASIS MRI dataset, which provides 3D brain scans along with their corresponding segmentation masks. Each mask contains four labels, representing different anatomical regions in the brain.

### Dataset Details

- Training set: 9664 samples.
- Validation set: 1120 samples.
- Test set: 544 samples.

<p align="center">
  <img src="./figures/case_441_slice_0.nii.png" width="45%" style="display: inline-block; margin-right: 5px;">
  <img src="./figures/seg_441_slice_0.nii.png" width="45%" style="display: inline-block;">
  <br>
  <b>Original MRI Slice (left) and Ground-Truth Segmentation Mask (right)</b>
</p>

### Dataset Preprocessing

#### Image preprocessing (MRI images)

- Convert images to tensors with values in `[0, 1]`.
- Normalize using a mean (`0.1317`) and standard deviation (`0.1864`) to stabilize training and improve convergence.

#### Mask preprocessing (segmentation labels)

- Convert masks to integer tensors (`int64`) representing class IDs.
- Map pixel values to class labels:
  - `0` -> class `0`
  - `85` -> class `1`
  - `170` -> class `2`
  - `255` -> class `3`
- Preserve masks as discrete labels without normalization.

## Model Architecture

![UNet Architecture](figures/u_net.webp)

U-Net is a U-shaped convolutional neural network for image segmentation, it consists of an encoder layer and an decoder layer.

- The encoder extracts important features through repeated convolution and downsampling. It uses max pooling (2×2 filters) to shrink the image size while keeping important information.

- The decoder upsamples to increase the image size to get back to the original image size. it also combines information from the encoder using “skip connections.” These connections help the decoder get spatial details that might have been lost when shrinking the image.

## Training Process

### Training Details

- Epochs: `15`.
- Batch size: `64`:
- Optimzer: Adam.
- Learning rate: `1e-4`.
- Weight decay: `5e-4`.
- Loss function: Cross-entropy loss.

### Evaluation Details

The Dice Coefficient is used to evaluate the model. The Dice coefficient is a similarity metric commonly used to evaluate image segmentation performance. It measures the overlap between the predicted segmentation and the ground truth mask, with values ranging from 0 to 1.

```
Dice Coefficient = 2 x area of intersection / area of union
```

## Training Result

### Test Set Dice Scores

| Class    | Dice Score |
| -------- | ---------- |
| 0        | 0.9981     |
| 1        | 0.9219     |
| 2        | 0.9466     |
| 3        | 0.9688     |
| **Mean** | **0.9588** |

### Image segmentation and ground-truth

![Test set results](figures/test_set_results.png)

## References

1. GeeksforGeeks. (n.d.). _U-Net architecture explained_. https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/

2. GeeksforGeeks. (2025, July 23). _What are different evaluation metrics used to evaluate image segmentation models?_ https://www.geeksforgeeks.org/computer-vision/what-are-different-evaluation-metrics-used-to-evaluate-image-segmentation-models/

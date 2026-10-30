# Milestone 2 — SurfaceGuard AI

## 1. Objective

Build a reliable segmentation data pipeline by decoding RLE annotations, exploring class distributions and defect co-occurrences, implementing a PyTorch Dataset, and training a baseline segmentation model.

## 2. Dataset

- Training images: 11,393 (verify against the loaded dataset)
- Annotation rows: 45,572 (verify against the loaded dataset)
- Image resolution: 128 × 800
- Defect classes: defect_a, defect_b, defect_c, defect_d
- Annotation format: 1-indexed column-major RLE

## 3. Exploratory Analysis

Include the class distribution, co-occurrence matrix, Pearson correlation matrix, and maximum pixel-level overlap.

## 4. PyTorch Dataset

The Dataset returns RGB images with shape (3, 128, 800) and four-channel segmentation masks with shape (4, 128, 800). Both are represented as float32 tensors.

## 5. Baseline Model

- Architecture: Three-stage U-Net
- Input channels: 3
- Output channels: 4
- Loss: BCEWithLogitsLoss
- Optimizer: AdamW
- Learning rate: 0.0003
- Batch size: 4
- Epochs: 5
- Validation split: 20%
- Random seed: 42

## 6. Training Results

Insert the complete five-epoch training and validation loss table and the loss plot from the notebook.

Report the best validation-loss epoch and discuss the observed generalization behaviour.

## 7. Conclusion

Milestone 2 establishes the RLE decoding, data loading, exploratory analysis, and baseline segmentation pipeline needed for subsequent model improvements.

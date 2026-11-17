# Milestone 3 Report — Transfer Learning & Stronger Segmentation Models

**Project:** SurfaceGuard AI — Industrial Surface Anomaly Segmentation  
**Course:** BSDA2001P — Introduction to DL and GenAI Project  
**Student:** Manandip Das

## 1. Objective
The objective of Milestone 3 is to explore transfer learning and stronger segmentation architectures for industrial surface defect detection.

## 2. Dataset and Configuration
- Input resolution: 128 × 800 grayscale images.
- Output classes: `defect_a`, `defect_b`, `defect_c`, `defect_d`.
- The same train-validation split from Milestone 2 is reused.
- Batch size: 16; epochs: 5; seed: 42.
- Optimizer: AdamW, learning rate: 0.0003, weight decay: 0.01.
- Loss: 0.5 BCE + 0.5 Soft Dice.
- Evaluation threshold: 0.5; precision: float32.

## 3. Questions and Key Findings
- **Q1:** Four stride-2 downsampling stages produce an 8 × 50 bottleneck, a one-pixel odd-width skip mismatch, and a downsampling factor of 256.
- **Q2:** Compared ResNet-34 and EfficientNet-B0 U-Net parameter counts and examined encoder freezing.
- **Q3:** Demonstrated that summing RGB convolution weights preserves grayscale convolution output, with a 3× input-memory reduction.
- **Q4:** Calculated effective receptive fields of 13 and 25 for dilation rates 6 and 12.
- **Q5:** Calculated Soft Dice losses of 0.9851 and 0.0524 for the two synthetic examples.
- **Q6:** Explained why softmax can assign different classes to different pixels but cannot represent overlapping labels at one pixel.
- **Q7:** Derived the Soft Dice gradient for empty targets and discussed its behaviour in mixed batches.
- **Q8:** Compared scratch and ImageNet-pretrained ResNet-34 U-Net models.
- **Q9:** Compared pretrained ResNet-34 and EfficientNet-B0 in terms of parameters, CUDA memory and validation performance.

## 4. Experimental Results
The Q8 and Q9 comparisons are based on the executed five-epoch experiments. Actual validation scores, generalization gaps, parameter counts and CUDA-memory measurements must be inserted from the notebook outputs.

## 5. Conclusion
Milestone 3 strengthened the understanding of transfer learning, encoder-decoder architectures, grayscale adaptation, receptive fields and segmentation losses. The experiments provide a basis for selecting a suitable pretrained backbone using validation performance, model size and memory requirements.

## 6. Reproducibility
Training histories, model checkpoints and comparison results are saved in the Kaggle working directory. The original M2 split and experiment configuration are retained for reproducibility.
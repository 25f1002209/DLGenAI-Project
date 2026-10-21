# Milestone 1 — SurfaceGuard AI

## 1. Objective

Explore the industrial surface-defect segmentation dataset, understand the rare defect classes and empty-mask imbalance, and establish a baseline for segmentation and threshold selection.

## 2. Baseline experiment

- **Model:** U-Net with a pretrained ResNet-34 encoder
- **Input:** Grayscale images at 128 × 800 resolution
- **Output:** Four sigmoid segmentation channels, one for each defect class
- **Training:** 5 epochs
- **Validation:** 80/20 image-level split using label-combination stratification
- **Thresholds evaluated:** 0.3, 0.5, 0.7, 0.9
- **Post-processing:** No minimum-area filtering

## 3. Threshold comparison

| Threshold | Naive Dice | Foreground Dice | FP pair rate | Assessment score |
|---:|---:|---:|---:|---:|
| 0.3 | 0.1875 | 0.4565 | 86.31% | 0.5297 |
| 0.5 | 0.2144 | 0.4559 | 83.09% | 0.5331 |
| 0.7 | 0.2504 | 0.4522 | 78.75% | **0.5354** |
| 0.9 | **0.3955** | 0.4238 | 60.98% | 0.5343 |

The validation split contained 84.19% empty image-class pairs.

## 4. Observations

Increasing the threshold reduced false-positive predictions, but also reduced foreground Dice. The naive Dice metric favored the 0.9 threshold, whereas the weighted assessment score was highest at 0.7.

The high proportion of empty image-class pairs makes empty-mask handling important when comparing thresholds. The results show why naive mean Dice alone may not be suitable for selecting the operating threshold.

## 5. Conclusion

A threshold of 0.7 achieved the best weighted assessment score in this experiment, while 0.3 achieved the highest foreground Dice. These results establish a baseline for future model and threshold improvements.

**Reproducibility:** See `notebooks/milestone_1_q11_q12.ipynb` for the experiment code, validation results, and threshold sweep. The assessment score is the notebook's weighted metric and is not a guarantee of the official Kaggle leaderboard score.

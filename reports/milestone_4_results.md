# Milestone 4 Report
## SurfaceGuard AI — Industrial Surface Defect Segmentation

### 1. Introduction

Milestone 4 focuses on understanding the data imbalance and training dynamics of the SurfaceGuard AI industrial surface defect segmentation task. The work examines image-level and pixel-level class imbalance, develops a bounded square-root sampling strategy, defines a warmup-plus-cosine learning-rate schedule, considers early stopping and GPU-memory constraints, and analyses why a naive Dice score can be misleading under the competition's official metric.

The dataset contains four defect classes: `defect_a`, `defect_b`, `defect_c`, and `defect_d`. Each image has dimensions of 128 × 800 pixels. The notebook uses the competition's run-length encoded masks and decodes them in column-major order.

### 2. Dataset and experimental subset

The training annotation file contains **45,572 annotation rows** corresponding to **11,393 unique images**. To create a fixed and reproducible subset, the notebook:

1. Sorts the unique image IDs.
2. Selects the first 1,200 IDs.
3. Retains all four class annotation rows for every selected image, resulting in **4,800 image-class rows**.
4. Groups images by their defect-class combination. Any combination occurring fewer than three times is mapped to `rare_other`.
5. Creates a stratified 80:20 image-level split using `random_state=42`.

The resulting split contains **960 training images** and **240 validation images**. The same split is intended to be used for the controlled experiments so that their results are comparable.

### 3. Q1 — Image-level and class-level imbalance

The selected subset contains:

| Category | Number of images |
|---|---:|
| Clean images (no annotated defect class) | 578 |
| Images containing `defect_a` | 107 |
| Images containing `defect_b` | 35 |
| Images containing `defect_c` | 438 |
| Images containing `defect_d` | 181 |

There are **15 distinct raw defect-presence combinations**. The largest combination group contains 578 images and the smallest contains one image.

The class counts are presence counts, not mutually exclusive image counts. An image containing multiple defect classes contributes to the count of each applicable class. Therefore, the class counts should not be added to estimate the number of defective images.

**Interpretation:** The subset is substantially imbalanced. Clean images form a large portion of the data, while `defect_b` is particularly under-represented at the image level. The imbalance motivates a sampling strategy that gives more exposure to less frequent defect combinations without allowing rare groups to dominate training.

### 4. Q2 — Pixel-level imbalance and `pos_weight` motivation

For `defect_b`, the notebook decodes the masks and calculates the number of positive pixels across all 1,200 selected images.

| Measurement | Result |
|---|---:|
| Positive `defect_b` pixels as a percentage of all `defect_b` mask pixels | 0.0160% |
| Mean `defect_b` area among positive images | 561.5 pixels |
| Negative-to-positive pixel ratio | 6,251:1 |

Only a very small fraction of the `defect_b` mask pixels are positive. This demonstrates that image-level imbalance is compounded by extreme pixel-level imbalance.

**Interpretation:** A segmentation model can obtain a deceptively low pixel-wise loss by predicting background for most pixels. Positive-class weighting in a binary cross-entropy objective can help counter this effect. The notebook quantifies the imbalance but does not report a final empirically tuned `pos_weight`; that value should be selected and validated during model training rather than assumed from the ratio alone.

### 5. Q3 — Bounded square-root sampling

The sampling weights are calculated using only the 960 training images. For each defect-combination group \(g\), the notebook uses:

\[
w_g = \sqrt{\frac{N_{\text{clean}}}{N_g}}
\]

where \(N_{\text{clean}}\) is the number of clean training images and \(N_g\) is the number of training images in group \(g\). Each image receives the weight of its group, and the weights are normalized to form sampling probabilities.

The observed clean-image fraction changes as follows:

| Quantity | Fraction |
|---|---:|
| Natural clean fraction in the training split | 0.48 |
| Expected clean fraction under weighted sampling | 0.27 |

The notebook assigns larger weights to rarer groups. For example, the weight for the `clean` group is 1.00, while the weight for the `defect_b` group is approximately 5.07. Several very rare or combined groups receive larger weights.

**Interpretation:** Square-root weighting reduces the dominance of clean images while being less aggressive than full inverse-frequency weighting. It is intended to improve exposure to rare defect combinations while limiting the risk of repeatedly oversampling a tiny number of examples. The reported 0.27 is an expected sampling fraction derived from the weights, not a measured fraction from a completed training epoch.

### 6. Q4 — Warmup and cosine learning-rate schedule

The notebook defines a step-based warmup-plus-cosine schedule with:

- Base learning rate: \(3 \times 10^{-4}\)
- Warmup duration: 60 optimizer steps
- Total schedule duration: 480 optimizer steps

During warmup, the learning rate increases linearly. After warmup, it follows a cosine decay.

| Optimizer step | Learning rate |
|---:|---:|
| 15 | \(7.500 \times 10^{-5}\) |
| 270 | \(1.500 \times 10^{-4}\) |
| 400 | \(2.606 \times 10^{-5}\) |

Because the schedule is defined in optimizer steps, `scheduler.step()` must be called after each optimizer update. Calling it only once per epoch would not follow the schedule as designed.

### 7. Q5 — Early stopping

The notebook provides an early-stopping trace with:

- Patience: 3 epochs
- Minimum improvement (`min_delta`): 0.002
- Best-checkpoint restoration: restore the epoch with the best validation score

However, the `scores` list is empty in the uploaded notebook. The notebook explicitly asks for the 12 validation scores from the milestone source table to be entered before executing the trace.

**Status:** The early-stopping policy is defined, but the stopping epoch, best epoch, and corresponding score cannot be reported from the current notebook execution. These values must be calculated after entering the source scores. A second trace with `min_delta=0` is also indicated for comparison.

### 8. Q6 — AMP, GPU memory and gradient accumulation

The notebook uses an illustrative memory estimate based on 25 million model parameters. It estimates 16 bytes per parameter for weights, gradients, and AdamW optimizer states:

\[
25{,}000{,}000 \times 16 / 10^6 = 400\text{ MB}
\]

Using the notebook's assumed activation-memory estimates and a 3,400 MB budget gives:

| Estimate | Result |
|---|---:|
| Weights, gradients and AdamW states | 400 MB |
| Batch 16, FP32 total estimate | 2,800 MB |
| Batch 16, AMP total estimate | 1,600 MB |
| Maximum estimated batch size, FP32 | 20 |
| Maximum estimated batch size, AMP | 40 |

The notebook also considers microbatching with a microbatch size of 8 and accumulation factor of 4, giving an effective batch size of 32. For 960 training images over 5 epochs, it estimates 120 microbatches per epoch and 30 optimizer steps per epoch. An 8% warmup therefore corresponds to 12 optimizer steps.

**Interpretation and limitation:** These are planning estimates based on assumed activation memory, not measured GPU-memory consumption. Actual memory use depends on the architecture, input pipeline, precision, optimizer implementation, and runtime overhead. AMP and gradient accumulation are proposed to make training more memory-efficient while preserving a larger effective batch size.

### 9. Q7 — Naive Dice and the official metric

The selected subset contains 4,800 image-class pairs: 1,200 images multiplied by four defect classes. Of these, **4,039 pairs are empty**, meaning that the corresponding image has no ground-truth pixels for that class.

The notebook demonstrates the all-empty prediction case:

| Metric calculation | Result |
|---|---:|
| Empty image-class pairs | 4,039 |
| Empty-pair fraction / naive mean Dice in the all-empty case | 0.84146 |
| Official score in the all-empty prediction example | 0.30 |

The example illustrates why averaging Dice over all image-class pairs can be misleading. A model predicting no defects can receive credit on empty pairs even though it fails to segment every foreground defect.

The notebook's official-metric illustration uses the weighted structure:

\[
S = 0.70D_{\text{foreground}}
  + 0.20S_{\text{false-positive}}
  + 0.10A_{\text{empty}}
\]

In the all-empty prediction example, foreground Dice is treated as zero, while the false-positive and empty-accuracy terms are one, producing:

\[
S = 0.70(0) + 0.20(1) + 0.10(1) = 0.30
\]

**Interpretation:** Evaluation should report foreground segmentation quality and false-positive behaviour, not only an aggregate Dice value dominated by empty pairs. The official metric implementation used for final model selection must match the competition's precise evaluation rules.

### 10. Q8 — Six-epoch training dynamics

The notebook prepares the dataset and training components for a six-epoch experiment:

- Images are loaded and converted from BGR to RGB.
- Images are resized to 128 × 800 if required.
- Four binary masks are reconstructed from the RLE annotations.
- Training augmentation includes horizontal flipping, vertical flipping, and mild brightness/contrast changes.
- Images are converted to floating-point tensors and scaled to the [0, 1] range.
- The loss combines binary cross-entropy with logits and Dice loss, each with a weight of 0.5.
- The validation metric reports foreground Dice, false-positive score, empty accuracy, and the weighted official score.

The training and validation DataLoaders are configured with batch size 16. However, the notebook does not contain a completed six-epoch training loop or resulting epoch-by-epoch validation trace. It also notes that the `ScratchUNet` architecture must be copied from or defined using the Milestone 3 model.

**Status:** The data pipeline, loss, metric functions, and loaders are prepared. Actual six-epoch loss curves, validation scores, best epoch, and training dynamics remain to be generated by completing and running the training loop.

### 11. Q9 — Defect-size bins and presence versus overlap

The notebook groups positive image-class masks into three area ranges:

| Defect area | Positive image-class pairs |
|---|---:|
| 1–100 pixels | 5 |
| 101–500 pixels | 99 |
| 501+ pixels | 657 |

The `501+` group accounts for **99.24% of all positive pixels** in the selected subset.

The notebook then applies illustrative miss-rate assumptions of 70%, 30%, and 5% to the three bins respectively. Under those assumptions, the estimated missed-pair counts are:

| Defect area | Positive pairs | Assumed miss rate | Expected missed pairs |
|---|---:|---:|---:|
| 1–100 | 5 | 70% | 3.50 |
| 101–500 | 99 | 30% | 29.70 |
| 501+ | 657 | 5% | 32.85 |
| **Total** | **761** | — | **66.05** |

The largest estimated missed-pair contribution is from the `501+` bin, because it contains the greatest number of positive pairs, even though its assumed miss rate is lowest.

**Interpretation:** Presence detection and mask overlap measure different aspects of performance. A model may correctly identify that a defect exists while producing a poor mask, or it may miss a small defect entirely. The size-bin analysis motivates reporting both defect-presence recall and overlap quality. The miss rates in this calculation are illustrative assumptions, not measured model recall values.

### 12. Q10 — Controlled one-factor-at-a-time experiments

The notebook specifies four six-epoch experiments using the same 960/240 split:

| Run | Seed | Sampler | Learning-rate schedule | AMP |
|---|---:|---|---|---|
| R0 | 42 | Natural | Constant | Yes |
| R0b | 7 | Natural | Constant | Yes |
| R1 | 42 | Natural | Warmup + cosine | Yes |
| R2 | 42 | Bounded square-root | Constant | Yes |

The purpose of the design is to isolate the effects of random seed, learning-rate scheduling, and sampling strategy:

- **R0 vs R0b:** assesses sensitivity to the random seed.
- **R0 vs R1:** isolates the effect of the warmup-plus-cosine schedule.
- **R0 vs R2:** isolates the effect of bounded square-root sampling.

For each run, the notebook asks for the best official validation score and the empty-pair false-alarm rate to be recorded from the actual W&B run.

**Status:** The experiment plan is present, but no run results are recorded in the uploaded notebook. Therefore, no claim can yet be made about which configuration performs best.

### 13. Overall findings

The executed analysis establishes several important properties of the selected data:

1. The 1,200-image subset is highly imbalanced at both image and pixel levels.
2. `defect_b` is particularly sparse, with only 35 positive images and 0.0160% positive pixels in its class masks.
3. A bounded square-root sampler is expected to reduce the clean-image share from 48% to 27% in sampled training batches.
4. A step-based warmup-plus-cosine schedule is defined and its learning-rate values are verified at selected optimizer steps.
5. Empty image-class pairs are common, making naive mean Dice unsuitable as the only model-selection metric.
6. Most positive pixels belong to the 501+ pixel defect-size group, but small defects remain important for presence recall.
7. AMP and gradient accumulation provide a practical memory-planning strategy, although the notebook's memory figures are estimates rather than hardware measurements.

### 14. Pending work and reproducibility notes

The following items are not completed in the uploaded notebook and should be filled using actual source values or executed experiments:

- **Q5:** Enter the 12 validation scores from the specified milestone source table and compute the early-stopping trace.
- **Q8:** Define/import `ScratchUNet`, complete the training loop, and record all six epochs of training and validation metrics.
- **Q10:** Execute R0, R0b, R1, and R2 for six epochs each; record the best official validation score and empty-pair false-alarm rate from W&B.
- If the memory estimates are to be presented as measured hardware results, profile actual peak allocated and reserved GPU memory during training.

All numerical findings in this report are taken from outputs present in the uploaded notebook. Where a result is not present, it is explicitly marked as pending rather than estimated or fabricated.

### 15. Conclusion

Milestone 4 provides a quantitative understanding of the data imbalance and establishes a controlled framework for improving segmentation training. The analysis supports the use of balanced sampling, a step-based learning-rate schedule, mixed precision, and an evaluation strategy that accounts for foreground defects and false positives. The completed data analysis and experiment design form the basis for the next training stage. Final conclusions about early stopping, six-epoch training dynamics, and the impact of sampling and scheduling must wait until the corresponding experiments have been executed and their actual validation results recorded.

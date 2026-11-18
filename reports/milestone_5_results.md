# Milestone 5 Report
## SurfaceGuard AI — Industrial Surface Defect Segmentation

## 1. Introduction

Milestone 5 converts the trained SurfaceGuard AI segmentation models into a competition-ready inference and submission pipeline. The milestone focuses on converting model outputs into binary masks, choosing class-specific thresholds, removing small false-positive components, applying flip-based test-time augmentation (TTA), combining predictions from multiple models or checkpoints, and generating a valid run-length encoded (RLE) submission within Kaggle's time and memory limits.

The task contains four output classes: defect_a, defect_b, defect_c, and defect_d. Images and masks use a native canvas of 128 × 800 pixels. The report follows the nine questions in the supplied Milestone 5 form and uses the uploaded notebook outputs where available. Where the notebook does not calculate the exact quantity requested by the form, that gap is explicitly identified rather than replaced with an unsupported number.

## 2. Q1 — Logits, probability thresholds and the strict-inequality trap

The form gives six logits: −2.0, −0.9, −0.5, 0.0, 0.8, 2.2, with ground truth [0, 0, 1, 1, 1, 0]. A logit z is converted to probability using the sigmoid function σ(z) = 1 / (1 + e^(−z)).

| Requested quantity | Correct result |
| --- | --- |
| Logit corresponding to T = 0.3 | ln(0.3 / 0.7) = −0.8473 |
| Positive count at T = 0.5 using p > T | 2 |
| Positive count at T = 0.5 using p ≥ T | 3 |
| Dice at T = 0.3 using strict > | 0.8571 |
| Dice at T = 0.5 using strict > | 0.4000 |

At T = 0.5, sigmoid(z) is greater than 0.5 exactly when z > 0. Thus only 0.8 and 2.2 are positive under the strict rule, whereas the non-strict rule also includes the zero logit. At T = 0.3, the probability condition σ(z) > 0.3 is equivalent to z > −0.8473. The four logits −0.5, 0.0, 0.8 and 2.2 are therefore predicted positive. This gives three true positives and one false positive, so Dice = 2TP / (2TP + FP + FN) = 6/7 = 0.8571.

Important notebook correction: the uploaded Q1 code uses +0.9 and +0.5 rather than the −0.9 and −0.5 shown in the supplied form. Its printed Dice values (0.7500 and 0.5714) therefore do not correspond to the question as written. The values in this report are recalculated from the form's logits.

## 3. Q2 — Per-class threshold and minimum-area selection

The official score is computed as Score = 0.70 × FgDice + 0.20 × FalsePositiveScore + 0.10 × EmptyAccuracy. Applying this formula to the supplied sweep table gives:

| Config | T | Min area | FgDice | FalsePositiveScore | EmptyAccuracy | Official score |
| --- | --- | --- | --- | --- | --- | --- |
| X | 0.5 | 0 | 0.62 | 0.9950 | 0.40 | 0.6730 |
| Y | 0.7 | 100 | 0.58 | 0.9990 | 0.85 | 0.6908 |
| Z | 0.9 | 300 | 0.45 | 0.9998 | 0.97 | 0.6120 |

Configuration Y should be selected because it has the highest official score, 0.6908. Configuration X has the highest foreground Dice (0.62), but it performs substantially worse on EmptyAccuracy (0.40) and slightly worse on FalsePositiveScore (0.9950). Y improves these two components to 0.85 and 0.9990. Configuration Z has the strongest false-positive and empty-pair terms, but its foreground Dice falls to 0.45; the loss in the heavily weighted foreground term makes its total score lower. This question demonstrates why choosing a threshold only by FgDice can select the wrong configuration.

## 4. Q3 — Connected components, connectivity and area filtering

The supplied 6 × 8 binary mask is analysed using OpenCV connected-component labelling. Under 8-connectivity, diagonal neighbours belong to the same component; under 4-connectivity, only horizontal and vertical neighbours are connected.

| Measurement | Result |
| --- | --- |
| Number of components under 8-connectivity | 3 |
| Number of components under 4-connectivity | 5 |
| Foreground pixels remaining with 8-connectivity and min area ≥ 5 | 11 |
| Foreground pixels remaining with 8-connectivity and min area ≥ 6 | 6 |

The count changes from five components to three because diagonal contacts merge regions under 8-connectivity. The minimum-area condition is inclusive in the implementation: a component is retained when its area is greater than or equal to the configured minimum. Raising the minimum from 5 to 6 removes components of area exactly 5, reducing the retained foreground from 11 pixels to 6 pixels.

## 5. Q4 — Does post-processing improve the official score?

The validation set contains 100 image-class pairs: 20 positive and 80 empty. Each mask has 128 × 800 = 102,400 pixels. Before filtering, the 20 positive pairs comprise 12 pairs with Dice 0.70 and 8 pairs with Dice 0.45. Among the empty pairs, 50 are completely empty, 20 contain a 20-pixel speckle, and 10 contain a 2,000-pixel blob. A minimum-area threshold N = 50 removes all 20 speckles and does not remove the 2,000-pixel blobs. It also removes the only predicted 30-pixel component from one positive pair whose Dice was 0.45.

| Metric | Before filtering | After filtering |
| --- | --- | --- |
| Foreground Dice | 0.6000 | 0.5775 |
| FalsePositiveScore | 0.9975 | 0.9976 |
| EmptyAccuracy | 0.6250 | 0.8750 |
| Official score | 0.6820 | 0.6913 |

Before filtering, mean foreground Dice = (12 × 0.70 + 8 × 0.45) / 20 = 0.6000. After filtering, one of the 0.45 positive pairs becomes zero, giving (12 × 0.70 + 7 × 0.45 + 1 × 0) / 20 = 0.5775.

The false-positive score is 1 minus the false-positive pixel count divided by the total pixels in the 80 empty pairs. Before filtering, false-positive pixels = 20 × 20 + 10 × 2,000 = 20,400, so the score is 0.9975. After filtering, only 10 × 2,000 = 20,000 false-positive pixels remain, giving 0.9976. EmptyAccuracy rises from 50/80 = 0.6250 to 70/80 = 0.8750 because the 20 speckle-containing pairs become completely empty.

Using the official weighted formula gives 0.6820 before filtering and 0.6913 after filtering. Therefore, the filter improves the total score by approximately 0.0093 despite reducing foreground Dice. The gain in EmptyAccuracy and FalsePositiveScore outweighs the loss in foreground overlap for this particular validation example. This does not mean that N = 50 is universally optimal; it must be selected on representative validation data.

## 6. Q5 — Recall ceiling of minimum-area filtering at pair level

The form asks for the number of positive image-class pairs that would become completely empty after applying N = 100, and for the defect class that loses the most pairs. A pair is wiped out only when every connected component in that pair has area strictly smaller than 100 pixels. Counting individual components smaller than 100 is not equivalent to counting wiped-out pairs: a pair may contain several small components, or a small component alongside a component that survives.

The uploaded notebook's Q5 cell currently reports component counts, not the pair-level quantity requested by the form. It reports 1,017 components smaller than 100 pixels (7.011% of all positive components). This is useful component-level evidence, but it cannot be submitted as the answer to the question's pair-level count, and it does not identify the class with the most wiped-out pairs.

To obtain the required answer, the notebook must group the RLE masks by image and class, run 8-connected component analysis on each positive pair, and count a pair as wiped out when no component has area ≥ 100. The following code performs that calculation using the existing decode_rle helper and train.csv:

```python
df = pd.read_csv(TRAIN_CSV, keep_default_na=False)
wiped = {c: 0 for c in CLASSES}
positive_pairs = {c: 0 for c in CLASSES}
for _, row in df.iterrows():
    cls = row["class_name"]  # use the actual class-column name in train.csv
    mask = decode_rle(row["mask_rle"])
    if not mask.any():
        continue
    positive_pairs[cls] += 1
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    if not np.any(component_areas >= 100):
        wiped[cls] += 1
print("Total positive pairs wiped out:", sum(wiped.values()))
print("Wiped-out pairs by class:", wiped)
print("Class losing the most pairs:", max(wiped, key=wiped.get))
```

Before running this code, confirm the class-column name in train.csv; if it is named differently, replace class_name accordingly. The exact total and most-affected class are not available in the uploaded notebook output and are therefore left as pending rather than guessed.

## 7. Q6 — Test-time augmentation and inverse-flip correctness

The model input layout is (B, C, H, W) = (1, 1, 128, 800). Therefore, horizontal flipping is performed on width dimension 3 and vertical flipping on height dimension 2. In PyTorch, the corresponding operations are torch.flip(x, [3]) and torch.flip(x, [2]).

The original 2 × 2 prediction is [[0.9, 0.6], [0.4, 0.2]]. The horizontal-flip output must be inverse-flipped left-to-right before averaging; the vertical-flip output must be inverse-flipped top-to-bottom. After inverse transformation, the three predictions are averaged pixel by pixel.

| Requested quantity | Correct result |
| --- | --- |
| Horizontal flip dimension | 3 |
| Vertical flip dimension | 2 |
| Averaged top-right probability | 0.6667 |
| Positive pixels at strict T = 0.5 after correct inverse-flips | 2 |
| Positive pixels if inverse-flips are forgotten | 1 |

The correctly aligned average is [[0.8000, 0.6667], [0.3000, 0.3333]]. Two pixels exceed 0.5. If the transformed predictions are averaged without undoing their flips, spatially different locations are mixed; only one pixel exceeds the strict threshold. This is the inverse-transform bug that the question highlights.

## 8. Q7 — Ensembling: soft averaging, hard voting and averaging space

Three models predict five pixels. The ground truth is [1, 0, 1, 1, 1], and all binary decisions use the strict threshold p > 0.5.

| Requested quantity | Result |
| --- | --- |
| Soft probability-average Dice | 0.8571 |
| Hard majority-vote Dice | 0.5714 |
| Weighted ensemble probability at p1 (weights 0.5, 0.3, 0.2) | 0.6600 |
| Probability-space average of logits +1 and −3 after sigmoid | 0.3892 |
| Logit-space average followed by sigmoid | 0.2689 |
| Decision at T = 0.30 | Probability average predicts positive; logit average predicts negative |

Soft averaging first averages the three probabilities at each pixel and then thresholds the result. It predicts [1, 0, 1, 0, 1], giving three true positives and one false negative, so Dice = 6/7 = 0.8571. Hard voting thresholds each model separately and takes a majority. It predicts [0, 1, 1, 0, 1], giving two true positives, one false positive and two false negatives, so Dice = 4/7 = 0.5714.

At p1, the weighted average is 0.5(0.90) + 0.3(0.40) + 0.2(0.45) = 0.6600. For logits +1 and −3, sigmoid probabilities are approximately 0.7311 and 0.0474. Their probability average is 0.3892. Averaging logits first gives −1, whose sigmoid is 0.2689. At threshold 0.30 the probability-space average is positive, while the logit-space average is negative. These approaches are not interchangeable because sigmoid is nonlinear.

## 9. Q8 — Inference budget: memory and time

The hidden test set contains 5,000 images. Each image has 4 output classes and a 128 × 800 map. The total number of stored map elements is 5,000 × 4 × 128 × 800 = 2,048,000,000.

| Storage format | Memory for all test probability maps |
| --- | --- |
| float32 (4 bytes/element) | 8.192 GB |
| float16 (2 bytes/element) | 4.096 GB |
| uint8 (1 byte/element) | 2.048 GB |

If every model/view prediction is retained as a separate float32 map, Config A (3 models × 3 views) requires 9 × 8.192 = 73.728 GB. Config B (2 models × 2 views) requires 4 × 8.192 = 32.768 GB. Neither fits within a 16 GB budget. If predictions are accumulated into one running float32 sum rather than all being stored, the full-map accumulator is 8.192 GB, which fits within 16 GB for either configuration (before accounting for model weights, activations, and other runtime allocations).

| Configuration | Forward passes | Pure model time at 12 ms/pass/image |
| --- | --- | --- |
| A: 3 models × 3-view TTA | 5,000 × 3 × 3 = 45,000 | 540 seconds = 9 minutes |
| B: 2 models × 2-view TTA | 5,000 × 2 × 2 = 20,000 | 240 seconds = 4 minutes |

The notebook limit is 30 minutes and data loading, RLE encoding and I/O consume 8 minutes, leaving 22 minutes for model inference. One complete 3-view model takes 3 minutes, so at most floor(22/3) = 7 models can be ensembled. One complete 2-view model takes 2 minutes, so at most floor(22/2) = 11 models can be ensembled. These are theoretical limits based on the supplied 12 ms per image forward-pass time; real throughput should be benchmarked because preprocessing, GPU contention and checkpoint loading add overhead.

## 10. Q9 — Designing the final inference notebook

The final inference notebook should implement the following pipeline in a fixed order:

- Read the test image and preserve its identifier and original spatial dimensions.
- Apply the same deterministic preprocessing used during training: image decoding, RGB conversion, resizing and normalization. Do not introduce training-only random augmentation.
- For each selected checkpoint/model, run the original image and the chosen flip views (for example, original, horizontal flip and vertical flip). Apply sigmoid to logits when combining probabilities.
- Inverse-transform every flipped prediction back to the original image orientation before any averaging. This is essential because pixels must refer to the same spatial locations.
- Average the aligned TTA probabilities for each model/checkpoint. Then combine model/checkpoint predictions using the selected soft or weighted probability ensemble.
- Apply the tuned threshold separately for each class using the strict comparison p > T_c.
- Run connected-component filtering separately for each class and remove components whose area is below that class's selected minimum area.
- Encode each final binary mask using the competition's RLE convention (column-major / Fortran order) and write the exact required sample-submission schema and row order.
## 11. Q9 — Why pipeline order matters

- Thresholding before TTA or model averaging discards probability information. Averaging already-binarized masks is a different operation and can change decisions near the threshold.
- Averaging flipped predictions before inverse-flipping combines different image coordinates and produces spatially incorrect masks.
- Applying component filtering before thresholding is undefined for a probability map unless a separate probability-based rule is specified; connected components operate on binary foreground regions.
- Encoding RLE before the final threshold and area filter creates a stale submission mask. Any later change to pixels requires re-encoding.
- Resizing, thresholding, filtering and RLE encoding must use a consistent spatial resolution. If masks are resized, the final mask must be restored to the required native 128 × 800 canvas before RLE encoding.
## 12. Q9 — Avoiding holdout overfitting and the all-empty baseline

- Use a separate calibration/validation split or cross-validation folds. Tune thresholds and minimum areas on calibration data, then report the selected configuration on a separate untouched validation set.
- Prefer a coarse, constrained search and stable per-class settings rather than an excessively large grid of thresholds and minimum areas. Check whether selected values remain similar across folds or random seeds.
- Do not repeatedly inspect the same holdout while making successive parameter changes. Repeated optimization gradually overfits the holdout even when each individual change appears small.
- Compare every post-processing configuration against an all-empty baseline. The baseline exposes how much score can be obtained from empty image-class pairs and prevents a high aggregate score from being mistaken for strong defect segmentation.
## 13. Q9 — Empty masks and submission safety checklist

An entirely empty binary mask should be encoded as an empty RLE string ("") when that is what the competition's sample_submission schema expects. It should not be replaced by an arbitrary run such as "0 0". The implementation's encode_rle helper returns an empty string for an all-zero mask.

- Check that the submission has exactly the same required columns as sample_submission.csv, with no extra index column.
- Check that every expected test image/class row appears exactly once and in the same order as sample_submission.csv; do not rely on filesystem iteration order.
- Check that RLE values contain valid start-length pairs, are within the 128 × 800 canvas, and decode back to masks of the expected shape.
- Check that empty masks are represented with the required empty-string convention and that no RLE entries are NaN or unintended nulls.
- Run a small end-to-end smoke test before the full inference run, including one image, all classes, TTA inversion, ensemble, thresholding, area filtering and RLE round-trip decoding.
- Estimate peak memory and inference time with the selected number of checkpoints and TTA views. Use running-sum accumulation, release temporary tensors, and avoid retaining all per-model/per-view maps.
- Confirm the output file exists at the required Kaggle working path, has the expected number of rows, and can be read back with pandas before submission.
## 14. Overall findings and limitations

The numerical exercises show that the official metric can favour a configuration with lower foreground Dice when it substantially improves false-positive control and empty-pair accuracy. Connected-component filtering can therefore improve the total score even when it removes a small true defect, but this trade-off must be measured on representative validation data.

Flip TTA is only correct when each transformed output is inverse-transformed before averaging. Soft probability ensembling outperforms hard majority voting on the supplied five-pixel example. Probability averaging and logit averaging can also produce different threshold decisions.

The full float32 probability-map tensor for the test set is 8.192 GB. Retaining every model/view output is infeasible under the 16 GB budget for both configurations, while a single running float32 accumulator is 8.192 GB and is the more memory-efficient approach. Under the supplied time assumptions, Config A takes 9 minutes and Config B takes 4 minutes for the forward passes.

Two source/notebook issues remain. First, the Q1 notebook input signs differ from the logits printed in the form; the report uses the form values and gives corrected answers. Second, the Q5 notebook cell counts small connected components, while the form asks for positive image-class pairs completely wiped out by the filter. The pair-level total and most-affected class must be obtained by running the corrected pair-level calculation against the competition train.csv. The uploaded notebook does not provide those values.

## 15. Conclusion

Milestone 5 establishes the final inference logic required to turn segmentation probabilities into a competition submission. The correct process is to align TTA predictions, ensemble probabilities, apply per-class thresholds, remove undersized connected components, and only then encode masks as RLE. The worked questions demonstrate the metric trade-offs introduced by post-processing, the importance of strict threshold conventions, and the memory/time advantages of streaming accumulation. Before final submission, the Q5 pair-level calculation should be corrected and run, and the complete inference notebook should be checked for schema, row-order, RLE and runtime integrity.

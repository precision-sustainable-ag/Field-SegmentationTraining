# Segmentation-mask comparison and grouping analysis

This directory contains two related analysis scripts:

- `compare_predicted_masks.py` compares binary predicted masks with binary
  ground-truth masks, calculates image-level and model-level metrics, joins the
  results to biological groupings, and optionally creates plots and mask grids.
- `analyze_mask_grouping_metrics.R` analyzes the grouping-expanded metrics,
  creates box plots, and uses one-way ANOVA followed by Tukey's Honestly
  Significant Difference (HSD) test to compare group or model means.

There is no `compare_predicted_mass.py` file in this directory. References to
that name mean `compare_predicted_masks.py`.

## Paths and expected inputs

Both scripts currently use this repository-relative dataset root:

```text
data/nav_seg_model_accuracy_comp_paper/data/test_dataset/
└── images_masks_grouped/all_images_masks_in_grouping/
```

`compare_predicted_masks.py` defines that location as `BASE_DIR` and expects:

```text
BASE_DIR/
├── gt_masks/
│   ├── image_001.png
│   └── ...
├── images/
│   ├── image_001.jpg
│   └── ...
└── predict_masks/
    ├── model_a/
    │   └── masks/
    │       ├── image_001.png
    │       └── ...
    └── model_b/
        └── masks/
            └── ...
```

The source photographs under `images/` are only needed for overlay figures.
The script searches for `.jpg`, `.jpeg`, and `.png` photographs, in that order.
Ground-truth and predicted masks are matched by filename stem (`image_001`, for
example), not by full path.

The grouping metadata is read from:

```text
data/nav_seg_model_accuracy_comp_paper/data/test_dataset/
└── balanced_groupings_test_images.json
```

The JSON may contain the grouping mapping at its top level or beneath a
`test_grouping` key. The expected structure is:

```json
{
  "test_grouping": {
    "grouping_type": {
      "grouping_name": ["path/to/image_001.jpg", "path/to/image_002.jpg"]
    }
  }
}
```

Here, a grouping type might be a classification scheme such as biological
group, family, or morphology, while a grouping is one category within that
scheme. Paths can include directories because only each filename stem is used
as the `image_id`.

## Running the scripts

Run the mask comparison first, from the repository root:

```bash
python scripts/compare_predicted_masks.py
```

Then run the R analysis after
`per_image_metrics_by_grouping.csv` has been created:

```bash
Rscript scripts/analyze_mask_grouping_metrics.R
```

The R script requires `ggplot2`, `dplyr`, `viridis`, `readr`, and
`multcompView`. It checks for all five packages before doing any analysis,
stops with a list of any that are missing, and does not install them. Although
`viridis` is loaded, the current black-box-plot implementation does not call a
Viridis color scale.

## `compare_predicted_masks.py`

### Purpose and processing flow

For every immediate model directory under `PRED_DIR`, the script indexes PNG
files inside its `masks/` directory and finds image IDs also present in
`GT_DIR`. Each matching pair is loaded as grayscale. A pixel value greater than
zero becomes foreground (`True` or class 1); zero becomes background (`False`
or class 0). The two masks must have identical dimensions. A mismatched pair is
logged and skipped.

The two-dimensional masks are flattened before they are passed to
TorchMetrics. Flattening changes only their shape, not the pixel
correspondence. Metrics are calculated independently for every valid
image/model pair.

The following switches control the pipeline:

| Switch | Effect |
|---|---|
| `CALCULATE_METRICS` | Calculates per-image metrics and model summaries. |
| `CREATE_GROUPING_METRICS` | Joins per-image results to the grouping JSON and writes grouping CSVs. |
| `MAKE_MASK_GRIDS` | Creates figures showing ground truth and predicted masks. |
| `MAKE_OVERLAY_GRIDS` | Creates figures overlaying masks on source photographs. |
| `MAKE_BAR_CHARTS` | Creates one model-level mean bar chart per metric. |

If metric calculation is disabled, the script attempts to load an existing
`per_image_metrics.csv`. Bar-chart creation similarly uses an existing
`summary_metrics.csv` when necessary. Grouping analysis therefore can be run
from previously calculated metrics.

### Confusion-matrix terms

All reported metrics are based on these pixel counts:

- **True positive (TP):** prediction and ground truth are both foreground.
- **False positive (FP):** prediction is foreground but ground truth is
  background.
- **False negative (FN):** prediction is background but ground truth is
  foreground.
- **True negative (TN):** prediction and ground truth are both background.

### TorchMetrics imports

The Python script imports these binary classification metrics from
`torchmetrics.classification`:

```python
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryAccuracy,
)
```

`BinaryJaccardIndex`, `BinaryF1Score`, `BinaryPrecision`, `BinaryRecall`, and
`BinaryAccuracy` are assembled into a TorchMetrics `MetricCollection` for the
foreground calculations. The Jaccard and F1 classes are also instantiated
separately for background calculations using inverted masks.

These inputs are already binary boolean masks. They are converted to integer
0/1 tensors before being passed to TorchMetrics, so class 1 is foreground and
class 0 is background. No probability threshold, ROC AUC, or precision-recall
AUC is calculated.

### Metric formulas

#### Foreground IoU / Jaccard index (`fg_iou`)

The Jaccard index and intersection over union (IoU) are the same metric. For
foreground pixels:

```text
fg_iou = TP / (TP + FP + FN)
```

The numerator is the intersection of predicted and true foreground. The
denominator is their union. It is calculated with `BinaryJaccardIndex` while
foreground is the positive class.

#### Foreground Dice / F1 score (`fg_dice`)

For binary masks, the Dice coefficient and F1 score are equivalent:

```text
fg_dice = 2 × TP / (2 × TP + FP + FN)
```

Equivalently, it is the harmonic mean of foreground precision and recall. It
is calculated with `BinaryF1Score`.

#### Precision (`precision`)

```text
precision = TP / (TP + FP)
```

Precision answers: of all pixels predicted as foreground, what fraction truly
are foreground? It is calculated with `BinaryPrecision`.

#### Recall (`recall`)

```text
recall = TP / (TP + FN)
```

Recall answers: of all true foreground pixels, what fraction did the model
identify? It is calculated with `BinaryRecall`.

#### Accuracy (`accuracy`)

```text
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Accuracy is the fraction of all pixels classified correctly. It is calculated
with `BinaryAccuracy`. Accuracy can appear high for masks dominated by
background, even when foreground performance is weak; foreground IoU and Dice
are therefore important complementary measures.

#### Background IoU

The script inverts the prediction and target with `~`, making original
background pixels the positive class, and applies `BinaryJaccardIndex` again:

```text
bg_iou = TN / (TN + FP + FN)
```

In this formula, TN is the correctly identified original background. The same
two original error regions, FP and FN, form the missed portion of the
background union. Background IoU is calculated internally but is not written
as its own CSV column.

#### Mean IoU (`mean_iou`)

The per-image mean IoU gives foreground and background equal class weight:

```text
mean_iou = (fg_iou + bg_iou) / 2
```

This is a class macro-average, not an IoU calculated by pooling all pixels from
all images.

#### Background Dice

The script also applies `BinaryF1Score` to the inverted masks:

```text
bg_dice = 2 × TN / (2 × TN + FP + FN)
```

Background Dice is calculated internally but is not written as its own CSV
column.

#### Mean Dice (`mean_dice`)

The per-image mean Dice gives foreground and background equal class weight:

```text
mean_dice = (fg_dice + bg_dice) / 2
```

If a foreground calculation fails and remains `NaN`, the current fallback uses
the available background score as the mean. Other caught TorchMetrics failures
also remain `NaN`. Pandas ignores `NaN` values when calculating summaries.

### Model-level averaging

The script first calculates all metrics separately for each image. It then
groups rows by model and takes the arithmetic mean of each metric:

```text
model metric = sum of valid per-image metric values / number of valid images
```

This is image-level macro averaging. Each image has equal influence regardless
of its dimensions or number of foreground pixels. It differs from concatenating
all pixels and calculating one global confusion matrix. `num_compared` is the
number of image rows for that model.

### Grouping metrics

When `CREATE_GROUPING_METRICS` is enabled, the JSON is converted to a long table
with `image_id`, `grouping_type`, and `grouping`. Duplicate memberships are
removed. This table is inner-joined to the per-image metrics:

- images absent from either side are omitted;
- an image can produce multiple rows if it belongs to multiple grouping types
  or groupings;
- no segmentation metric is recalculated during this join.

The grouping summary averages the per-image values within each combination of
`model`, `grouping_type`, and `grouping`. Its `num_compared` column counts the
joined rows contributing to each combination.

### Python outputs

All artifacts are saved under:

```text
BASE_DIR/mask_comparison_metrics/
```

The enabled switches determine what is created:

```text
mask_comparison_metrics/
├── per_image_metrics.csv
├── summary_metrics.csv
├── per_image_metrics_by_grouping.csv
├── summary_metrics_by_grouping.csv
├── bar_charts/
│   ├── fg_iou.png
│   ├── fg_dice.png
│   ├── precision.png
│   ├── recall.png
│   ├── mean_iou.png
│   ├── mean_dice.png
│   └── accuracy.png
├── mask_grids/
│   └── <image_id>_masks.png
└── overlay_grids/
    └── <image_id>_overlays.png
```

- `per_image_metrics.csv` contains one row per valid image/model pair.
- `summary_metrics.csv` contains one row per model with image-level means.
- `per_image_metrics_by_grouping.csv` contains the grouping-expanded image
  rows used by the R script.
- `summary_metrics_by_grouping.csv` contains means and counts for every model,
  grouping type, and grouping combination.
- `bar_charts/` contains model-summary charts for metrics having valid values.
- `mask_grids/` shows ground truth and available model masks. Titles display
  foreground IoU when it is available.
- `overlay_grids/` shows the same masks as neon-green overlays. If a source
  photograph is missing, the script uses a black background.

Existing files with the same names are overwritten.

## `analyze_mask_grouping_metrics.R`

### Purpose

The R analysis asks two kinds of questions:

1. Within each model, does segmentation performance differ among categories of
   a particular grouping type, such as different families or morphologies?
2. Across the rows in the grouping-expanded data, do the models have different
   overall metric distributions?

It reads:

```text
BASE_DIR/mask_comparison_metrics/per_image_metrics_by_grouping.csv
```

Here, `BASE_DIR` is hard-coded in the script as:

```text
data/nav_seg_model_accuracy_comp_paper/data/test_dataset/
└── images_masks_grouped/all_images_masks_in_grouping/
```

The input path, output path, metric candidates, and significance level are
script constants; the script has no command-line options. Because the paths
are repository-relative, run it from the repository root unless the constants
are changed.

Required columns are `model`, `image_id`, `grouping_type`, and `grouping`. It
analyzes whichever of `fg_iou`, `fg_dice`, `precision`, `recall`, `mean_iou`,
`mean_dice`, and `accuracy` are present.

### Validation and summary

The script stops with an informative error if packages, the input file,
required identifier columns, or all candidate metric columns are missing. It
creates the output directory recursively, reads the CSV with
`readr::read_csv(show_col_types = FALSE)`, and converts `model`,
`grouping_type`, and `grouping` to factors.

It then calculates an arithmetic mean for every available metric and a row
count (`n`) for every model/grouping-type/grouping combination. Missing metric
values are ignored separately by each mean through `na.rm = TRUE`. The `n`
column is the total number of input rows in the combination, not the number of
non-missing observations for each metric. Consequently, a metric that is
missing for every row in a combination can have a nonzero `n` and a non-finite
mean.

### Grouping-wise box plots

For every available metric and every grouping type, the script:

1. selects non-missing values for that metric and grouping type;
2. draws metric distributions by grouping as black box plots;
3. facets the figure so each model gets its own panel;
4. runs a separate one-way ANOVA and Tukey HSD analysis inside each model;
5. converts pairwise adjusted p-values to compact-letter labels;
6. remaps labels so the category with the highest observed mean receives `a`;
7. places the letters at the bottom of each model panel; and
8. saves a 12 × 7 inch, 150-DPI PNG.

Grouping types are processed in sorted character order. Within each plot,
factor levels retain the order produced when the full input columns were
converted to factors; the script does not reorder grouping categories by their
means. Facets use a common, fixed y-axis scale.

The boxes, outlines, and outliers are black. Boxes use 0.7 alpha, width 0.65,
and line width 0.35. The custom minimal theme uses a pale blue plot and panel
background, light-blue major grid lines, no minor grid lines, bold dark-blue
facet labels and centered title, and x-axis labels rotated 90 degrees. Compact
letters are bold dark blue, size 3.8.

Y-axis limits are calculated from finite observed values, padded by 12%,
with at least 0.01 padding, bounded to 0 through 1, and expanded toward a
minimum displayed span of 0.05. If no finite values exist, the fallback range
is 0 to 1. Constant-valued data use the greater of 0.01 or 1% of the absolute
value as initial padding. The script uses `coord_cartesian` with no scale
expansion and clipping disabled. These limits affect presentation only, not
the statistical tests. Letters are positioned 3% of the displayed y-span
above its lower limit.

Rows with `NA` for the plotted metric are removed before plotting. The
statistical helper additionally keeps only finite numeric values, so infinite
values do not enter ANOVA or Tukey HSD. A grouping plot is skipped if no rows
or grouping levels remain.

### What one-way ANOVA does

A one-way analysis of variance (ANOVA) evaluates the null hypothesis that all
category population means are equal for one metric. For example, within one
model and grouping type, it fits:

```r
aov(fg_iou ~ grouping, data = model_data)
```

ANOVA compares variability among category means with variability among
observations inside categories. A sufficiently large ratio provides evidence
that at least one mean differs, but ANOVA alone does not identify which pair or
pairs differ.

The usual interpretation assumes independent observations, approximately
normal residuals within groups, and similar variances across groups. Metrics
bounded between 0 and 1 can violate normality or equal-variance assumptions,
especially near 0 or 1 or with small samples. The script does not run residual,
normality, equal-variance, or independence diagnostics.

### How Tukey's HSD works

After fitting the one-way ANOVA, the script calls:

```r
TukeyHSD(fit, "grouping")
```

Tukey's HSD compares every pair of factor levels while controlling the
family-wise error rate across that collection of comparisons. Each comparison
reports a mean difference, confidence interval, and multiplicity-adjusted
p-value. The script uses the adjusted p-values and `alpha_sig = 0.05`.

The script does not first inspect the omnibus ANOVA p-value to decide whether
to continue; it attempts Tukey HSD whenever the ANOVA model can be fit. It also
does not write the ANOVA tables, Tukey comparison tables, confidence intervals,
or numeric p-values to disk. Their pairwise decisions are represented only by
the compact letters drawn on the plots.

This adjustment matters because performing many unadjusted tests increases the
chance of finding at least one apparently significant difference by chance.
Tukey's method makes the family of pairwise conclusions more conservative and
is designed for all-pairs mean comparisons after an ANOVA model.

### What a p-value means

A p-value is the probability, assuming the relevant null hypothesis and model
assumptions are true, of obtaining a test result at least as incompatible with
the null hypothesis as the observed result. It is **not** the probability that
the null hypothesis is true, the probability that a result occurred by chance,
or a measure of the size or practical importance of a difference.

For a Tukey pairwise comparison in this script:

- adjusted `p < 0.05` is treated as evidence that the two population means
  differ;
- adjusted `p >= 0.05` means the analysis did not detect a difference at that
  threshold—it does not prove the means are equal;
- effect sizes, confidence intervals, sample sizes, distributions, and domain
  relevance should also be considered.

### Compact-letter displays

`multcompView::multcompLetters` converts the matrix of Tukey decisions into
letters:

- categories sharing at least one letter are not significantly different at
  the selected threshold;
- categories sharing no letter are significantly different;
- a category can have multiple letters when it overlaps more than one
  nonsignificant set.

The script renames the generated symbols in descending order of observed mean,
so `a` contains the highest-mean category, followed by `b`, and so forth. A
letter is a significance-group label, not a performance grade. Receiving `a`
does not by itself mean that a category is significantly better than every
other category; the shared-letter relationships determine that.

If there are fewer than two usable observations or fewer than two factor
levels, or if ANOVA, Tukey HSD, adjusted-p-value extraction, or letter creation
fails, the script skips the letters for that panel rather than stopping the
whole analysis.

The script accepts either `p adj` or `p.adj` as the adjusted-p-value column
returned by Tukey HSD. It supplies lowercase and uppercase letters to
`multcompLetters`; if more distinct significance symbols must be remapped than
that pool supports, it stops with an error. Multi-letter labels are preserved
and their component symbols are put into the remapped letter order.

### Overall model comparison

For each metric, the script combines all non-missing rows, orders models from
highest to lowest observed mean, fits:

```r
aov(metric_value ~ model, data = overall_data)
```

It applies Tukey HSD across models, creates the same compact-letter display,
and draws one box plot per model. Models are reordered from highest to lowest
observed mean for each metric, so their x-axis order can differ between plots.
The overall plots use the same dimensions, resolution, dynamic y-axis logic,
box styling, theme, letter styling, and bottom-of-plot letter placement as the
grouping-wise plots. An overall plot is skipped when no non-missing rows remain
or fewer than two model levels remain.

Important: the source is `per_image_metrics_by_grouping.csv`, an expanded join.
Consequently, the same image/model metric can occur multiple times when an
image belongs to multiple grouping types or categories. The overall model plot
and ANOVA operate on those rows as written; they do not deduplicate by
`model + image_id`. This can weight images with more memberships more heavily
and can treat repeated measurements as though they were separate observations.
For a strictly one-row-per-image overall comparison, the analysis would need to
use `per_image_metrics.csv` or explicitly deduplicate before fitting. Also,
because the same test images are evaluated by every model, a paired or repeated-
measures analysis may better represent model dependence than the current
one-way ANOVA, depending on the scientific question.

### R outputs

Everything is written under:

```text
BASE_DIR/mask_comparison_metrics/r_model_performance_across_grouping/
```

The layout is:

```text
r_model_performance_across_grouping/
├── grouping_metric_means.csv
├── fg_iou/
│   └── boxplot_<grouping_type>.png
├── fg_dice/
│   └── boxplot_<grouping_type>.png
├── precision/
│   └── boxplot_<grouping_type>.png
├── recall/
│   └── boxplot_<grouping_type>.png
├── mean_iou/
│   └── boxplot_<grouping_type>.png
├── mean_dice/
│   └── boxplot_<grouping_type>.png
├── accuracy/
│   └── boxplot_<grouping_type>.png
└── overall_model_comparison/
    ├── boxplot_model_fg_iou.png
    ├── boxplot_model_fg_dice.png
    ├── boxplot_model_precision.png
    ├── boxplot_model_recall.png
    ├── boxplot_model_mean_iou.png
    ├── boxplot_model_mean_dice.png
    └── boxplot_model_accuracy.png
```

Only directories and figures for metric columns present in the input are
created. Grouping-type names are made filename-safe by replacing unsupported
characters with underscores, trimming leading and trailing underscores, and
using `unnamed` if nothing remains. Metric names go through the same sanitizer
when directory or overall-plot filenames are constructed. Different original
names that sanitize to the same string can target the same filename. Existing
CSV and PNG files with the same names are overwritten; stale files for metrics
or grouping types no longer present are not removed.

At successful completion, the script prints the output directory. It does not
save fitted ANOVA objects, set a random seed, or modify the source CSV. All
calculations are deterministic for a fixed input and package environment.

## End-to-end data flow

```text
ground-truth masks + model masks
                │
                ▼
compare_predicted_masks.py
                │
                ├── per-image metrics
                ├── model mean metrics
                ├── grouping-expanded metrics ◄── grouping JSON
                └── optional charts and mask figures
                              │
                              ▼
              analyze_mask_grouping_metrics.R
                              │
                              ├── grouping means
                              ├── grouping-wise box plots
                              ├── within-model ANOVA + Tukey HSD letters
                              └── overall model box plots + Tukey letters
```

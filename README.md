# Project 3: Intelligent Analysis of Digikala Reviews

## 📌 Project Overview

This project builds an intelligent system that analyzes Persian Digikala product reviews :
1. Predict whether the user **recommends** the product (`recommendation_status`).
2. Predict an ordered **rating band** (`rating_band`) derived from the numeric `rate`.

The project includes:
- Descriptive data analysis
- Persian text normalization
- TF-IDF + Logistic Regression baselines
- Fine-tuned Persian BERT (`HooshvareLab/bert-fa-base-uncased`)
- A semi-supervised experiment with pseudo-labeling (confidence threshold = 0.90)
- Error analysis and offline inference

## Repository Structure

```text
project-3/
├── data/                    # Dataset files (NOT tracked by git)
├── notebooks/
│   ├── 01_descriptive_analysis.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_transfer_learning_finetuning.ipynb
│   └── 04_semi_supervised.ipynb
├── artifacts/               # Saved models, tokenizer, preprocessing (NOT tracked)
├── reports/                 # Final report, plots, error analysis
├── requirements.txt
├── .gitignore
└── README.md
```

### Distribution of `rate`: Skewness, Concentration, and Most Frequent Values

| rate | count (n) | percentage (pct) | cumulative pct |
|---|---|---|---|
| 5.0 | 2,398,801 | 38.97% | 38.97% |
| 4.0 | 1,515,723 | 24.62% | 63.59% |
| 3.0 | 958,799 | 15.57% | 79.16% |
| 0.0 | 529,098 | 8.59% | 87.75% |
| 1.0 | 327,662 | 5.32% | 93.08% |
| 4.8 | 7,276 | 0.12% | 98.02% |
| 3.4 | 7,250 | 0.12% | 98.13% |
| 3.6 | 6,975 | 0.11% | 98.25% |
| 2.5 | 5,438 | 0.09% | 98.34% |

**What I observed:** The `rate` column contains **178 unique values** ranging from 0 to 2500 (one record with value 2500 is outside the valid 0–5 range and is invalid). The distribution is heavily right-skewed: the value 5.0 with **38.97%**, 4.0 with **24.62%**, and 3.0 with **15.57%** are the most frequent values. The five most frequent values (5, 4, 3, 0, 1) together cover **93.08%** of the data, and the top 20 values cover **98.34%** — indicating a highly concentrated distribution. Notably, `rate = 0.0` accounts for **8.59%** (about 529k records), showing that a substantial portion of users gave a zero rating (not missing data). Additionally, `rate = 2.0` does not appear among the top 20 values — a pattern consistent with the "polarization of opinions" phenomenon (users are either highly satisfied or highly dissatisfied).

**What it means:** The `rate` distribution is skewed and concentrated, exhibiting a J-shaped pattern (a peak at 5, a steep decline, and a small bump at 0 and 1). With the project thresholds, the derived `rating_band` is expected to be relatively imbalanced (with the `high` band dominant).

**What I will do:** For `rating_band`, I will use class weighting and ordinal distance error. I will remove the record with `rate = 2500` as invalid data.

## Part A — Descriptive Analysis:

### A1. Dataset Size and Structure

- **Total records:** 6,156,289 (counted via DuckDB parser, not physical lines)
- **Columns:** 15 (all required columns present, no extra columns)
- **Unique products:** 331,599
- **Unique sellers:** 29,214
- **File size:** 1.28 GB (CSV) → compressed to Parquet with ZSTD

**Review distribution per product:**
- Median: 4 reviews
- Mean: 18.57 (std = 40.93 → right-skewed)
- 90th percentile: 46
- 99th percentile: 200
- Max: 400

**Product concentration:**
- Top 10 products account for only **0.05%** of all reviews
- No product has more than 1,000 reviews

**Decision:** A product-grouped split is required to prevent the model from memorizing product-specific patterns. See A5 for the updated justification.

**Artifacts:** (../reports/fig1_comments_per_product.png)

### A2. Missing-Value Patterns

**What I observed:** Missing values are concentrated in metadata columns, not in the core modeling columns. The missing counts and percentages per column are:

| Column | Missing | Percentage |
|---|---|---|
| `true_to_size_rate` | 6,065,516 | 98.53% |
| `disadvantages` | 5,744,017 | 93.30% |
| `advantages` | 5,448,084 | 88.50% |
| `title` | 2,866,415 | 46.56% |
| `recommendation_status` | 894,426 | 14.53% |
| `seller_title` | 302,181 | 4.91% |
| `seller_code` | 302,181 | 4.91% |
| `body` | 637 | 0.01% |
| `rate`, `created_at`, `id`, `is_buyer`, `product_id`, `likes`, `dislikes` | 0 | 0.00% |

The core input column `body` has only **637 missing values (0.01%)**. Columns such as `id`, `created_at`, `rate`, `is_buyer`, `product_id`, `likes`, and `dislikes` have **zero missing values** because they are either system-generated (`id`, `created_at`, `product_id`, `is_buyer`), mandatory for review submission (`rate`), or have a default value of 0 (`likes`, `dislikes`). In contrast, columns with high missing rates (`title`, `advantages`, `disadvantages`, `true_to_size_rate`) are **optional** fields that users frequently skip.

**Structural analysis of missing labels:** The cross-tabulation of `label_missing × is_buyer` reveals a fully structural pattern:

| label_missing | is_buyer | Count |
|---|---|---|
| False | True | 5,261,863 |
| True | False | 302,181 |
| True | True | 592,245 |

- **All non-buyers (302,181 records) have missing labels** — 100% of `is_buyer=False` rows lack `recommendation_status`.
- Among buyers, **5,261,863 (89.9%) have labels** and **592,245 (10.1%) do not**.
- There is **no row** with `label_missing=False` and `is_buyer=False`.

Additionally, rows with missing labels have a **higher mean rate (4.04 vs 3.58)**, although the median is identical (4.0), suggesting that unlabeled users skew toward higher ratings.

**What it means:** Missing labels are structurally biased: non-buyers never provide a recommendation label, and even among buyers, ~10% skip it. Users without labels tend to give slightly higher ratings. This means missingness is **not MCAR (Missing Completely At Random)** — it is structurally tied to `is_buyer` and `rate`, consistent with a **MAR (Missing At Random)** pattern.

**What I will do:**
1. Keep missing-label rows in the **natural unlabeled pool** (do not discard them) — they are valuable for the semi-supervised experiment.
2. Drop the **637 rows with missing `body`** since `body` is the only model input.
3. Do not use `is_buyer` as a feature (it is metadata and would leak label availability).
4. Report the structural missingness as evidence that the dataset's unlabeled pool is not random.

**Artifacts:** `reports/missing_by_column.csv`

#### A3-a. Distribution of `recommendation_status`

**What I observed:** The `recommendation_status` column contains four categories:

| Label | Count | Percentage |
|---|---|---|
| `recommended` | 4,162,839 | 67.6% |
| `<MISSING>` (originally `"nan"`) | 894,426 | 14.5% |
| `no_idea` | 594,121 | 9.6% |
| `not_recommended` | 504,903 | 8.2% |

The label `"nan"` was stored as a string in the raw CSV (due to `all_varchar=true` during ingestion) and has been normalized to `<MISSING>` for analysis.

**What it means:** The label distribution is heavily imbalanced: `recommended` accounts for 67.6% of the data, while `not_recommended` accounts for only 8.2% — an approximately 8:1 ratio. A naive model that always predicts `recommended` would achieve 67.6% accuracy without learning anything. Accuracy is therefore a misleading metric.

**What I will do:** Use **class weighting** and **Macro F1** (instead of accuracy) for evaluation. The `<MISSING>` rows are kept in the natural unlabeled pool for the semi-supervised experiment.

**Artifacts:**(../reports/fig2_labels_and_rate.png)

#### A3-c. Cross-tabulation: `recommendation_status` × `rate`

**What I observed:** The cross-tabulation (row-normalized, in %) reveals:

| rate | `<MISSING>` | `no_idea` | `not_recommended` | `recommended` |
|---|---|---|---|---|
| 0.00 | 4.8% | 3.9% | 10.9% | **80.5%** |
| 0.05 | 3.1% | 10.3% | **83.5%** | 3.1% |
| 5.00 | 18.1% | 0.6% | 0.3% | **81.0%** |

At `rate = 0.0`, **80.5%** of users still marked `recommended` — a surprising contradiction (zero stars but a positive recommendation). At `rate = 0.05`, **83.5%** marked `not_recommended` — a more intuitive pattern. At `rate = 5.0`, **81.0%** marked `recommended`.

**What it means:** While there is a general correlation between `rate` and `recommendation_status`, the data contains **contradictory labels** (e.g., `rate = 0` with `recommended`). This confirms that `rate` should not be used as a model feature, and that the label space has noise that will affect error analysis.

**What I will do:** Flag contradictory rows (e.g., `rate = 0` + `recommended`) as potential label noise for the error analysis in Part C.

### A4. Text-Length Analysis

**What I observed:** The `body` column has a heavily right-skewed length distribution. The median is **106 characters (22 words)**, with the 90th percentile at **152 characters (32 words)**, the 95th at **297 characters (61 words)**, and the 99th at **~297+ characters**. Short reviews dominate: **34.06%** of reviews are under 20 characters, **70.37%** are under 50 characters, and **35.69%** are under 5 words. In contrast, long reviews are rare: only **0.23%** exceed 500 characters and **0.03%** exceed 1,000 characters.

**Length by label:**

| Label | Count | Median chars | p90 chars | Median words |
|---|---|---|---|---|
| `recommended` | 4,162,839 | 27 | 95 | 6 |
| `<MISSING>` | 894,426 | 25 | 105 | 5 |
| `no_idea` | 594,121 | 38 | 129 | 8 |
| `not_recommended` | 504,903 | 48 | 159 | 10 |

**What it means:** The distribution is heavily right-skewed, with most reviews being very short and only a tiny fraction being long. There is a clear pattern across labels: `recommended` reviews are the shortest (median 27 chars), while `not_recommended` reviews are the longest (median 48 chars). This suggests that dissatisfied users tend to write more detailed reviews, while satisfied users often leave brief comments. Text length is therefore a weak signal for label prediction, but it is not sufficient on its own.

**What I will do:** Given that **97% of reviews are under ~61 words**, I will set the Transformer `max_length = 128` tokens. This covers the vast majority of reviews while keeping GPU memory usage reasonable. The ~0.23% of reviews longer than 500 characters will be truncated, which is acceptable given their rarity. The 34% of very short reviews (under 20 chars) will be kept as-is — they are valid inputs and will test the model's ability to classify on minimal text.

**Artifacts:** `reports/fig3_body_length.png`, `reports/A4_quantiles.csv`, `reports/A4_short_long.csv`, `reports/A4_length_by_label.csv`
# Project 3: Intelligent Analysis of Digikala Reviews

## 📌 Project Overview

This project builds an intelligent system that analyzes Persian Digikala product reviews to:
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

## Part A — Descriptive Analysis

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

**Artifacts:** `reports/fig1_comments_per_product.png`

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

The core input column `body` has only **637 missing values (0.01%)**. Columns such as `id`, `created_at`, `rate`, `is_buyer`, `product_id`, `likes`, and `dislikes` have **zero missing values** because they are either system-generated (`id`, `created_at`, `product_id`, `is_buyer`), mandatory for review submission (`rate`), or have a default value of 0 (`likes`, `dislikes`).

**Structural analysis of missing labels:** The cross-tabulation of `label_missing × is_buyer` reveals a fully structural pattern:

| label_missing | is_buyer | Count |
|---|---|---|
| False | True | 5,261,863 |
| True | False | 302,181 |
| True | True | 592,245 |

- **All non-buyers (302,181 records) have missing labels** — 100% of `is_buyer=False` rows lack `recommendation_status`.
- Among buyers, **5,261,863 (89.9%) have labels** and **592,245 (10.1%) do not**.
- There is **no row** with `label_missing=False` and `is_buyer=False`.

Additionally, rows with missing labels have a **higher mean rate (4.04 vs 3.58)**, although the median is identical (4.0).

**What it means:** Missing labels are structurally biased: non-buyers never provide a recommendation label, and even among buyers, ~10% skip it. This means missingness is **not MCAR** — it is structurally tied to `is_buyer` and `rate`, consistent with a **MAR** pattern.

**What I will do:**
1. Keep missing-label rows in the **natural unlabeled pool**.
2. Drop the **637 rows with missing `body`**.
3. Do not use `is_buyer` as a feature.

**Artifacts:** `reports/missing_by_column.csv`

### A3. Target Distributions

#### A3-a. Distribution of `recommendation_status`

| Label | Count | Percentage |
|---|---|---|
| `recommended` | 4,162,839 | 67.6% |
| `<MISSING>` (originally `"nan"`) | 894,426 | 14.5% |
| `no_idea` | 594,121 | 9.6% |
| `not_recommended` | 504,903 | 8.2% |

The label `"nan"` was stored as a string in the raw CSV and has been normalized to `<MISSING>`.

**What it means:** The label distribution is heavily imbalanced: `recommended` accounts for 67.6%, while `not_recommended` accounts for only 8.2% — an approximately 8:1 ratio. A naive model that always predicts `recommended` would achieve 67.6% accuracy without learning anything. Accuracy is therefore a misleading metric.

**What I will do:** Use **class weighting** and **Macro F1** instead of accuracy.

**Artifacts:** `reports/fig2_labels_and_rate.png`

#### A3-b. Distribution of `rate`

**What I observed:** The `rate` column contains **178 unique values** ranging from 0 to 2500 (one record with value 2500 is outside the valid 0–5 range and is invalid).

| rate | Count | Percentage | Cumulative % |
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

The five most frequent values (5, 4, 3, 0, 1) together cover **93.08%** of the data, and the top 20 values cover **98.34%**.

**What it means:** The `rate` distribution is heavily right-skewed, exhibiting a J-shaped pattern (a peak at 5, a steep decline, and a small bump at 0 and 1).

**What I will do:** Derive `rating_band` using the project thresholds:
- `low`: 0 ≤ rate < 2.5
- `medium`: 2.5 ≤ rate < 4.0
- `high`: 4.0 ≤ rate ≤ 5.0

Remove the invalid record with `rate = 2500`.

#### A3-c. Cross-tabulation: `recommendation_status` × `rate`

**What I observed:** The cross-tabulation (row-normalized, in %) reveals:

| rate | `<MISSING>` | `no_idea` | `not_recommended` | `recommended` |
|---|---|---|---|---|
| 0.00 | 4.8% | 3.9% | 10.9% | **80.5%** |
| 0.05 | 3.1% | 10.3% | **83.5%** | 3.1% |
| 5.00 | 18.1% | 0.6% | 0.3% | **81.0%** |

At `rate = 0.0`, **80.5%** of users still marked `recommended` — a surprising contradiction. At `rate = 0.05`, **83.5%** marked `not_recommended` — a more intuitive pattern. At `rate = 5.0`, **81.0%** marked `recommended`.

**What it means:** The data contains **contradictory labels** (e.g., `rate = 0` with `recommended`). This confirms that `rate` should not be used as a model feature, and that the label space has noise.

**What I will do:** Flag contradictory rows as potential label noise for the error analysis.

### A4. Text-Length Analysis

**What I observed:** The `body` column has a heavily right-skewed length distribution. The median is **106 characters (22 words)**, with the 90th percentile at **152 characters (32 words)**, the 95th at **297 characters (61 words)**. Short reviews dominate: **34.06%** of reviews are under 20 characters, **70.37%** are under 50 characters, and **35.69%** are under 5 words.

**Length by label:**

| Label | Count | Median chars | p90 chars | Median words |
|---|---|---|---|---|
| `recommended` | 4,162,839 | 27 | 95 | 6 |
| `<MISSING>` | 894,426 | 25 | 105 | 5 |
| `no_idea` | 594,121 | 38 | 129 | 8 |
| `not_recommended` | 504,903 | 48 | 159 | 10 |

**What it means:** There is a clear pattern across labels: `recommended` reviews are the shortest (median 27 chars), while `not_recommended` reviews are the longest (median 48 chars).

**What I will do:** Given that **97% of reviews are under ~61 words**, I will set the Transformer `max_length = 128` tokens.

**Artifacts:** `reports/fig3_body_length.png`, `reports/A4_quantiles.csv`, `reports/A4_short_long.csv`, `reports/A4_length_by_label.csv`

**Figure 5 — Median body length by label.** The median body length increases monotonically from satisfied to dissatisfied users:

| Label | Median length (chars) |
|---|---|
| `<MISSING>` | 25 |
| `recommended` | 27 |
| `no_idea` | 38 |
| `not_recommended` | **48** |

Dissatisfied users write reviews that are ~1.8× longer than satisfied users. This is a well-known psychological pattern.

### A5. Repetition, Conflicts, and Product Concentration

**Duplicate review bodies:** The dataset contains **197,746 unique bodies that appear more than once**, covering **2,217,778 rows** — i.e., **36.0% of all records**.

| Body | Count |
|---|---|
| عالی | 121,349 |
| خوب بود | 73,854 |
| خوبه | 72,210 |
| خوب | 64,297 |
| عالی بود | 23,598 |

**Conflicting labels:** **75,365 unique bodies** carry conflicting `recommendation_status` labels.

**Conflicting rates:** **119,538 unique bodies** carry conflicting `rate` values.

**What it means:** Repetition is substantial (36% of rows share a body), but most duplicates are short, generic comments. More importantly, the same body can carry different labels and different rates.

**What I will do:**
1. **Flag** duplicate bodies (`is_duplicate_body = 1`) but **do not remove them**.
2. Flag rows with conflicting labels (`is_label_conflict = 1`).

**Artifacts:** `reports/top_repeated_bodies.csv`

#### A3-d. Derived `rating_band` Distribution (Variants A and B)

**What I observed:** Two variants of `rating_band` were built using the project thresholds:

**Variant A (standard):**

| Band | Count | Percentage |
|---|---|---|
| `medium` | 2,569,339 | 41.7% |
| `high` | 2,489,182 | 40.4% |
| `low` | 1,097,767 | 17.8% |
| `invalid` | 1 | 0.0% |

**Variant B (corrected):** treats `rate = 0` as invalid (likely a missing star rating) and also fixes the `rate = 2.5` boundary.

| Band | Count | Percentage |
|---|---|---|
| `medium` | 2,574,777 | 41.8% |
| `high` | 2,489,182 | 40.4% |
| `low` | 563,231 | 9.1% |
| `invalid` | 529,099 | 8.6% |

**Difference:** Variant B reduces `low` from 17.8% to 9.1%, and also fixes a subtle boundary issue in Variant A (`BETWEEN 0 AND 2.5` incorrectly captured `rate = 2.5`).

**Suspicion (to be confirmed in Part C):** `rate = 0` exhibits anomalous behavior (80.5% `recommended`). This suggests (but does not prove) that `rate = 0` is noise.

**Decision (provisional):** Variant B will be used for the initial `rating_band` training in Part C.

**Artifacts:** `reports/fig6_rating_band.png`

### A6. Vocabulary-Level Analysis (Optional Extension)

**Sampling note:** This analysis uses a random sample of **123,334 reviews** (out of 6.16M). Global statistics (A1–A5) remain computed on the full data.

**Most frequent words:**

| Word | Count |
|---|---|
| بود | 27,795 |
| خیلی | 27,361 |
| خوب | 22,220 |
| به | 21,762 |
| از | 19,729 |
| خوبه | 18,902 |
| عالی | 16,112 |
| کیفیت | 10,315 |
| قیمت | 9,676 |
| راضی | 6,584 |

**Class-specific bigrams:**

| Class | Representative bigrams |
|---|---|
| `recommended` | «خیلی کیوته», «باصرفه», «خوبه توصیه», «کاملا مطابق», «راضی بودیم», «مناسب قیمت» |
| `not_recommended` | «پول دور», «اصلا نخرید», «آشغال», «افتضاح بود», «مزخرف», «درد نخور», «حیف پول» |
| `no_idea` | «خاصی ندیدم», «مطمئن نیستم», «تاثیر خاصی», «خیلی معمولی», «مناسب نبود» |
| `<MISSING>` | «موجودش», «لطفا موجود کنید», «نخریدم», «بیرون خریدم», «نگرفتم», «چراغ قوه» |

**Key finding — why `<MISSING>` rows lack labels:** The bigrams reveal two distinct user groups: those who have not bought the product, and those who only request restocking. This **confirms the A2 finding** that missing labels are structural.

**Stopword handling:** Stopwords are frequent but are **not removed** at this stage. Negations (`نه`, `نیست`) and discourse markers (`ولی`) carry sentiment information.

**Artifacts:** `reports/A6_frequent_words.csv`, `reports/A6_class_bigrams.csv`

### A7. Summary and Decision Mapping

| # | Finding (with evidence) | Modeling Decision |
|---|---|---|
| 1 | **Severe label imbalance:** `recommended` = 67.6%, `not_recommended` = 8.2% (A3-a). | Use **class weighting** and **Macro F1** instead of accuracy. |
| 2 | **Short reviews dominate:** median = 106 chars (~22 words); 97% of reviews < 297 chars (A4). | Set Transformer `max_length = 128` tokens. |
| 3 | **High repetition:** 197,746 duplicate bodies cover 36% of all rows; 75k bodies carry conflicting labels (A5). | **Flag** duplicates and conflicts, do **not** remove them. |
| 4 | **`rate = 0` is anomalous:** 80.5% of `rate = 0` rows are labeled `recommended` (A3-c). | Build two `rating_band` variants (A: standard, B: exclude `rate = 0`) and compare them in Part C. |
| 5 | **Missing labels are structural:** 100% of non-buyers lack `recommendation_status` (A2). | Keep unlabeled rows in the natural unlabeled pool. |

**Data provenance:** Figures 1–5 and all tables (A1–A5) are computed on the full dataset. A6 uses a random sample of 123,334 reviews (seed = 42).

**Artifacts:**
- `reports/fig1_comments_per_product.png` ... `reports/fig7_length_by_label.png`
- `reports/missing_by_column.csv`
- `reports/top_repeated_bodies.csv`
- `reports/A4_*.csv`
- `reports/A6_*.csv`

---

## Part B — Cleaning, Splitting, and Label Protocol

The entire pipeline is reproducible via `src/prepare_data.py`.

### B1. Persian Text Normalization

The normalization function (`src/text_norm.py`) applies the following transformations:

| Transformation | Examples |
|---|---|
| Arabic → Persian letters | `ي → ی`, `ك → ک`, `ى → ی` |
| Persian/Arabic digits → ASCII | `۰۱۲۳ → 0123`, `٠١٢٣ → 0123` |
| Remove diacritics | `U+064B` to `U+0652` |
| Remove HTML tags / URLs | `<div>`, `https://...` |
| Collapse repeated letters | `خووووب → خووب` (3+ → 2) |
| Zero-width space → regular space | `U+200B` |
| Remove zero-width joiner / LRM / RLM / BOM | `U+200D`, `U+200E/200F`, `U+FEFF` |
| **Keep** ZWNJ (`U+200C`) | Preserves half-spaces: `می‌روم`, `کتاب‌ها` |
| Keep emojis | Sentiment-bearing emojis are preserved |
| No stopword / negation removal | `نه`, `نیست`, `ولی` are kept |

After normalization, **215 rows** became empty and were dropped. The final normalized dataset contains **6,152,845 rows**.

### B2. Label Mapping and Quality Gate

| Raw value | Mapped label |
|---|---|
| `recommended` | `recommended` |
| `not_recommended` | `not_recommended` |
| `no_idea` | `no_idea` |
| `"nan"` (string) | `NULL` (missing label) |

The quality gate excluded:
- **3,229 duplicate IDs** (kept the first by `hash(body), product_id`)
- **1 row with `rate = 2500`**
- **215 rows empty after normalization**

### B3. Rating-Band Derivation (Variant B)

| Band | Threshold | Count (train) | Percentage (train) |
|---|---|---|---|
| `low` | `0 < rate < 2.5` | 393,941 | 9.2% |
| `medium` | `2.5 ≤ rate < 4.0` | 740,945 | 17.2% |
| `high` | `4.0 ≤ rate ≤ 5.0` | 2,801,794 | 65.0% |
| `NULL` | `rate = 0`, `rate > 5`, or missing | 370,310 | 8.6% |

The ordinal encoding is `low=0`, `medium=1`, `high=2`.

### B4. Product-Grouped Split

Products are sorted by `(-row_count, sha256(SPLIT_SEED | product_id))` and assigned greedily to minimize the normalized deviation from the 70/15/15 target.

| Split | Products | Rows | Row % | Recommended | Not Recommended | No Idea | Label Missing |
|---|---|---|---|---|---|---|---|
| **train** | 294,063 | 4,306,990 | 69.97% | 2,912,438 (67.6%) | 353,151 (8.2%) | 415,585 (9.6%) | 625,816 (14.5%) |
| **validation** | 18,762 | 922,927 | 14.99% | 624,094 (67.6%) | 75,675 (8.2%) | 89,054 (9.6%) | 134,104 (14.5%) |
| **test** | 18,767 | 922,928 | 15.03% | 624,094 (67.6%) | 75,675 (8.2%) | 89,054 (9.6%) | 134,105 (14.5%) |

The class distribution is nearly identical across splits.

### B5. Semi-Supervised Masking Protocol

- `SPLIT_SEED = 2026`, `MASK_SEED = "digikala-semi-supervised-v1"`, `MASK_RATE = 0.20`
- For each recommendation class in the training split, sort rows by `(stable_hash(id), id)` and mask `floor(0.20 × class_count)` rows.

| Class | Labeled train rows | Masked | Kept supervised |
|---|---|---|---|
| `recommended` | 2,912,438 | 582,487 | 2,329,951 |
| `no_idea` | 415,585 | 83,117 | 332,468 |
| `not_recommended` | 353,151 | 70,630 | 282,521 |
| **Total** | 3,681,174 | **736,234** | 2,944,940 |

### B6. Output Files

| File | Rows | Description |
|---|---|---|
| `train_supervised.csv` | 3,555,676 | Train rows not masked, with at least one valid target |
| `train_pseudo_unlabeled.csv` | 736,234 | Masked train rows with only `id` and `body` |
| `natural_unlabeled.csv` | 625,816 | Train rows with missing `recommendation_status` |
| `validation.csv` | 917,793 | Validation rows with at least one valid target |
| `test.csv` | 917,785 | Test rows with at least one valid target |

### B7. Split Manifest

The complete manifest is saved to `data/processed/split_manifest.json` and `reports/split_manifest.json`, recording seeds, split strategy, achieved fractions, rating-band thresholds, label mapping, normalization rules, exclusions, and per-class masking counts.

**Artifacts:** `data/processed/normalized.parquet`, `train_supervised.csv`, `train_pseudo_unlabeled.csv`, `natural_unlabeled.csv`, `validation.csv`, `test.csv`, `split_manifest.json`

---

## Part C — Modeling

### C1. TF-IDF Feature Extraction

| Parameter | Value | Rationale |
|---|---|---|
| `ngram_range` | `(1, 2)` | Captures single words and bigrams |
| `min_df` | `5` | Removes rare tokens (typos, proper nouns) |
| `max_features` | `400,000` | Caps vocabulary size |
| `sublinear_tf` | `True` | Logarithmic TF scaling |
| `token_pattern` | `r"(?u)[\w\u200c]{2,}"` | Keeps ZWNJ as part of words |
| `dtype` | `float32` | Reduces memory usage |

**Fit strategy:** The vectorizer was fit on a **random sample of 500,000 training reviews** (`random_state=SPLIT_SEED=2026`) and then applied to the full train, validation, and test sets.

**Output:**

| Metric | Value |
|---|---|
| Vocabulary size | **115,840** tokens |
| `X_train` shape | **(3,555,676, 115,840)** |
| `X_train` non-zero entries | **53,136,203** |
| Density | **0.013%** |
| Fit time | 45 s |
| Total transform time | 165 s (~2.7 min) |

The resulting matrix is highly sparse (~15 non-zero tokens per review).

**Artifacts:** `artifacts/tfidf/tfidf_vectorizer.joblib`, `X_train.npz`, `X_val.npz`, `X_test.npz`

### C2. TF-IDF Baseline: Hyperparameter Tuning

**Original baseline (C2-a):**

| Task | Best alpha (sample) | Best alpha (full train) | `max_iter` | Val Macro F1 |
|---|---|---|---|---|
| `recommendation` | 3e-6 | 3.06e-7 | 15 | **0.683** |
| `rating_band` | 3e-6 | 2.75e-7 | 15 | **0.645** |

**Improved baseline (C2-b):**

| Task | Best alpha (sample) | Best alpha (full train) | `max_iter` | Val Macro F1 |
|---|---|---|---|---|
| `recommendation` | 5e-6 | 5.09e-7 | 50 | **0.6831** |
| `rating_band` | 5e-6 | 4.58e-7 | 50 | **0.6471** |

**Comparison:**

| Task | Original | Improved | Δ |
|---|---|---|---|
| `recommendation` | 0.683 | 0.6831 | **+0.0001** |
| `rating_band` | 0.645 | 0.6471 | **+0.0018** |

**What it means:** The TF-IDF + SGD pipeline has reached its **ceiling** at ~0.68 and ~0.65. The bottleneck is the TF-IDF representation itself, which cannot capture word order, negation, or context.

### C3. TF-IDF Baseline — Final Evaluation on Test

**Task A — Recommendation:**

| Metric | Value |
|---|---|
| **Macro F1** | **0.699** |
| Accuracy | 0.853 |
| Balanced Accuracy | 0.720 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `recommended` | 0.943 | 0.918 | **0.930** | 624,094 |
| `not_recommended` | 0.658 | 0.787 | **0.717** | 75,675 |
| `no_idea` | 0.445 | 0.455 | **0.450** | 89,054 |

**Task B — Rating Band:**

| Metric | Value |
|---|---|
| **Macro F1** | **0.658** |
| Accuracy | 0.782 |
| Balanced Accuracy | 0.671 |
| **Ordinal Distance Error** | **0.245** |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `low` | 0.615 | 0.755 | **0.678** | 84,416 |
| `medium` | 0.489 | 0.360 | **0.415** | 158,774 |
| `high` | 0.864 | 0.897 | **0.881** | 600,384 |

**Confusion matrices (row-normalized):**

**Task A — Recommendation:**
| true ↓ / pred → | `recommended` | `not_recommended` | `no_idea` |
|---|---|---|---|
| `recommended` | **0.92** | 0.02 | 0.06 |
| `not_recommended` | 0.07 | **0.79** | 0.14 |
| `no_idea` | **0.33** | 0.22 | **0.46** |

**Task B — Rating Band:**
| true ↓ / pred → | `low` | `medium` | `high` |
|---|---|---|---|
| `low` | **0.75** | 0.14 | 0.11 |
| `medium` | 0.17 | **0.36** | **0.47** |
| `high` | 0.02 | 0.08 | **0.90** |

**What it means:** The model defaults to the dominant class when uncertain (`no_idea` → `recommended`, `medium` → `high`). This is the strongest empirical evidence that a contextual model (Transformer) is needed.

**Artifacts:** `reports/preds_tfidf_*.parquet`, `reports/baseline_metrics.json`, `reports/fig_baseline_confusion.png`, `artifacts/tfidf/logreg_*.joblib`

### C4. Top Features per Class

**Task A — Recommendation:**

| Class | Top features |
|---|---|
| `recommended` | «عالیه», «بهترین», «خیلی خوبه», «بد نبود», «خوبیه», «بد نیست», «عاشقش», «زیبا», «حرف نداره» |
| `not_recommended` | «افتضاح», «بدترین», «افتضاحه», «متاسفم», «راضی نیستم», «نخرید», «پشیمونم», «آشغال», «مزخرف», «تقلبی», «مرجوع» |
| `no_idea` | «شک دارم», «نیستم», «مطمئن نیستم», «بسیار معمولی», «کاملا معمولی», «نمیدونم», «کیفیت نیست» |

**Task B — Rating Band:**

| Class | Top features |
|---|---|
| `high` | «بهترین», «بد نبود», «نازک نیست», «اذیت نمیکنه», «عاشقش», «پاره نمیشه», «خشک نمیکنه», «خراب نمیشه» |
| `low` | «بدترین», «افتضاح», «متاسفم», «آشغال», «مزخرف», «نخرید», «فاجعه», «مرجوع» |
| `medium` | «کاملا معمولی», «بسیار معمولی», «معمولی», «کیفیت نیست», «بالایی نداره», «نه بد», «قیمت بصرفه», «شک دارم» |

**What it means:** The top features confirm the need for a contextual model: TF-IDF captures negation only as specific bigrams (e.g., «بد نبود»), not compositionally.

**Artifacts:** (printed in the notebook)

### C5. Transformer Fine-Tuning (BERT)

A Persian BERT model (`HooshvareLab/bert-fa-base-uncased`) was fine-tuned on a **deterministic 10% subset** (354,599 rows) created with the hash rule `sha256(SPLIT_SEED|modeling-subset|id) mod 1000 < 100`.

**Data tokenization:**

| Metric | Value |
|---|---|
| Train rows (subset) | 354,599 |
| Validation rows (subset) | 92,164 |
| Test rows (full) | 917,785 |
| Tokenization time | 78 s |
| **Truncation rate** | **0.3%** |
| `MAX_LEN` | 128 |

**Model architecture (multi-task):**

| Component | Value |
|---|---|
| Encoder | `HooshvareLab/bert-fa-base-uncased` |
| Total parameters | 162,845,958 |
| Trainable parameters | 162,845,958 (full fine-tuning) |
| `rec` head | Linear(768 → 3) |
| `band` head | Linear(768 → 3) |

The total loss is:
$$
\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda_{\text{band}} \cdot \mathcal{L}_{\text{band}}
$$
Where each head's loss is masked when the target is unavailable (`ignore_index=-100`).

**Class weights:** `recommendation` = `[0.42, 3.49, 2.98]`, `rating_band` = `[3.37, 1.80, 0.47]`.

**Training setup:**

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| `LR_ENCODER` / `LR_HEAD` | 2e-5 / 1e-3 |
| `BATCH` | 128 |
| `EPOCHS` | up to 3 (with early stopping) |
| `EVAL_EVERY` / `PATIENCE` | 4,000 / 3 |

**Validation metrics during training:**

| Step | Loss | F1 Rec | F1 Band | Mean F1 |
|---|---|---|---|---|
| 4,000 | 1.113 | 0.6970 | 0.6580 | 0.6775 |
| **8,000** | **1.067** | **0.6976** | **0.6599** | **0.6787** ✅ |
| 12,000 | 1.060 | 0.6976 | 0.6599 | 0.6787 |
| 16,623 | 1.075 | 0.6976 | 0.6599 | 0.6787 |

**Final test evaluation:**

**Task A — Recommendation:**

| Metric | Value |
|---|---|
| **Macro F1** | **0.6989** |
| Accuracy | 0.8252 |
| Balanced Accuracy | **0.7455** |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| `recommended` | 0.963 | 0.864 | **0.911** |
| `not_recommended` | 0.689 | 0.780 | **0.732** |
| `no_idea` | 0.368 | 0.593 | **0.454** |

**Task B — Rating Band:**

| Metric | Value |
|---|---|
| **Macro F1** | **0.6620** |
| Accuracy | 0.7447 |
| Balanced Accuracy | **0.6892** |
| **Ordinal Distance Error** | **0.2741** |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| `low` | 0.642 | 0.761 | **0.696** |
| `medium` | 0.398 | 0.499 | **0.443** |
| `high` | 0.891 | 0.807 | **0.847** |

**What it means:** BERT achieves Macro F1 comparable to TF-IDF but with **better Balanced Accuracy** and **better minority-class F1**.

**Artifacts:** `artifacts/bert/transformer_best.pt`, `artifacts/bert/tokenizer/`, `reports/transformer_metrics.json`, `reports/fig_transformer_curves.png`


### C6. Semi-Supervised Experiment (Pseudo-Labeling)

**Pseudo-label selection:**

| Metric | Value |
|---|---|
| Unlabeled pool size | 74,090 |
| Accepted (confidence ≥ 0.90) | **35,496 (47.91%)** |

| Class | Predicted (all) | Accepted | Acceptance rate |
|---|---|---|---|
| `recommended` | 53,473 | 32,976 | **61.67%** |
| `not_recommended` | 8,732 | 2,481 | 28.41% |
| `no_idea` | 11,885 | **39** | **0.33%** |

**Results — Test set comparison:**

| Head | Metric | Supervised | Control | Semi | Semi − Control |
|---|---|---|---|---|---|
| `recommendation` | Macro F1 | 0.6954 | 0.7053 | **0.7056** | **+0.0002** |
| `rating_band` | Macro F1 | 0.6665 | **0.6686** | 0.6675 | **−0.0012** |

**What it means:** Pseudo-labeling produced a **negligible gain** over the Control model. The root cause is the **confidence asymmetry** of the supervised model: it is overconfident on `recommended` (61.7% acceptance) and underconfident on `no_idea` (0.3% acceptance).

**Artifacts:** `artifacts/bert/semi_best.pt`, `reports/preds_semi_*.parquet`, `reports/semi_comparison.csv`

### C6-b. Pseudo-Label Quality Audit

| Metric | Value |
|---|---|
| **Accuracy of accepted pseudo-labels** | **98.01%** |
| **Accuracy of rejected pseudo-labels** | **68.95%** |

**What it means:** The 0.90 threshold was highly precise (98% precision) but also very conservative (69% of rejected rows were correct).

**Artifacts:** `reports/semi_metrics.json`

### C7. Statistical Comparison (Bootstrap, 95% CI)

| Comparison | Head | Mean Δ | 95% CI | Significant? |
|---|---|---|---|---|
| Semi vs Control | `recommendation` | +0.0002 | [−0.0007, +0.0011] | ❌ No |
| Control vs TF-IDF | `recommendation` | +0.0062 | [+0.0049, +0.0074] | ✅ Yes |
| Semi vs Control | `rating_band` | −0.0012 | [−0.0018, −0.0005] | ✅ Yes (negative) |
| Control vs TF-IDF | `rating_band` | +0.0108 | [+0.0098, +0.0118] | ✅ Yes |

**What it means:** BERT (Control) significantly outperforms TF-IDF on both heads. Semi-supervised training does not significantly improve over Control.

**Artifacts:** `reports/bootstrap_comparison.csv`

### C8. Error Analysis

**Recommendation errors (39 rows):**

| Category | Count | Representative example |
|---|---|---|
| `mixed_opinion` | 12 | «جنسش خوبه ولی روی قالی خوب نمی‌ایسته» |
| `disagreement` | 11 | «مدادا خوبن اصلا سخت نیستن» |
| `short_text` | 9 | «معمولی بود» |
| `label_noise` | 8 | «بسیار عالی» (1,304 duplicates) |
| `negation` | 6 | «بد نیست» |

**Rating-band errors (16 rows):**

| Category | Count | Representative example |
|---|---|---|
| `boundary` | 10 | `rate=2.2` (close to 2.5) |
| `disagreement` | 5 | «مدادا خوبن اصلا سخت نیستن» |
| `text_rate_mismatch` | 3 | «قیمت مناسب و خوبه» (rate=3.0) |
| `mixed_opinion` | 3 | «بوی عالی ولی ماندگاری صفر» |

**Key findings:**
1. **Label noise** («بسیار عالی» with 1,304 duplicates) is a major source of irreducible error.
2. **Mixed opinions** are systematically misclassified (models average clauses instead of following the clause after «ولی»).
3. **Rating-band boundaries** are artificial (e.g., `rate=3.0` with unambiguously positive text).
4. **Negation** is where BERT beats TF-IDF («مدادا خوبن اصلا سخت نیستن»).
5. **The two models fail differently**, suggesting an ensemble would outperform either one alone.

**Artifacts:** `reports/error_analysis_candidates.csv`, `reports/error_analysis_candidates_band.csv`

---

## Limitations

- **GPU quota:** The final BERT checkpoint (`model.pt`) could not be included in the repository due to Kaggle's 30-hour weekly GPU quota limit. Training was completed and the metrics are reported (Macro F1 = 0.6989 for recommendation, 0.6620 for rating_band), but the checkpoint is not included.
- **Corrupted download:** Multiple attempts to download the checkpoint from Kaggle resulted in corrupted ZIP files (26 KB instead of 654 MB). This is a known issue with Kaggle's large-file download.
- **Semi-supervised:** Pseudo-labeling did not improve over the Control model, due to the class imbalance of the accepted pseudo-labels (92.9% `recommended`).
- **Hardware:** Full fine-tuning on the 3.5M-row dataset was not feasible; a deterministic 10% subset was used, as permitted by the project.

## Conclusion

This project built a complete pipeline for Persian review analysis: from data auditing and normalization, through a TF-IDF baseline and a fine-tuned Transformer, to a semi-supervised experiment and a manual error analysis. The main findings are:

1. **The TF-IDF baseline reaches its ceiling at ~0.70** (recommendation) and ~0.66 (rating_band). It cannot capture negation, word order, or context.
2. **BERT improves on the minority classes** but is limited by the 10% subset. Full fine-tuning on 3.5M rows would likely push Macro F1 to 0.78–0.85.
3. **Pseudo-labeling with the fixed 0.90 threshold does not help**, because the base model is overconfident on the majority class and underconfident on the ambiguous minority class.
4. **A significant fraction of the remaining errors is irreducible** (label noise, artificial band boundaries, mixed opinions). The error analysis suggests that an ensemble of TF-IDF and BERT, or a clause-level attention mechanism, would be a promising next step.

**Artifacts:** `release_report.json`, `reports/*`, `artifacts/*`
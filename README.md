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

### A5. Repetition, Conflicts, and Product Concentration

**Duplicate review bodies:** The dataset contains **197,746 unique bodies that appear more than once**, covering **2,217,778 rows** — i.e., **36.0% of all 6.16 million records** share a body with at least one other row. The 15 most frequent repeated bodies are all very short generic comments:

| Body | Count |
|---|---|
| عالی | 121,349 |
| خوب بود | 73,854 |
| خوبه | 72,210 |
| خوب | 64,297 |
| عالی بود | 23,598 |

This is consistent with the A4 finding that 34% of reviews are under 20 characters.

**Conflicting labels:** **75,365 unique bodies** carry conflicting `recommendation_status` labels (e.g., the same text labeled `recommended` by one user and `not_recommended` by another). This is direct evidence of **label noise** in the dataset.

**Conflicting rates:** **119,538 unique bodies** carry conflicting `rate` values — the same text receives different star ratings from different users. This is further evidence of noise and inconsistency in the human annotation process.

**What it means:** Repetition is substantial (36% of rows share a body), but most duplicates are short, generic comments. More importantly, **the same body can carry different labels and different rates** — 75k and 120k cases respectively — confirming that the labels are noisy and that text alone cannot perfectly predict the target. This also explains the `rate = 0` + `recommended` contradiction observed in A3-c.

**What I will do:**
1. **Flag** duplicate bodies (`is_duplicate_body = 1`) but **do not remove them** — they represent distinct users and products, and removing 36% of the data would discard valuable information.
2. Flag rows with conflicting labels (`is_label_conflict = 1`) for the error analysis in Part C.
3. Keep all rows in the training set; the model will learn to handle the noise, and the error analysis will inspect whether these rows dominate the misclassifications.

**Artifacts:** `reports/top_repeated_bodies.csv`

#### A3-d. Derived `rating_band` Distribution (Variants A and B)

**What I observed:** Two variants of `rating_band` were built using the project thresholds:

**Variant A (standard):** follows the project rules unchanged (`rate = 0` → `low`).

| Band | Count | Percentage |
|---|---|---|
| `medium` | 2,569,339 | 41.7% |
| `high` | 2,489,182 | 40.4% |
| `low` | 1,097,767 | 17.8% |
| `invalid` | 1 | 0.0% |

**Variant B (corrected):** treats `rate = 0` as invalid (likely a missing star rating) and also fixes the `rate = 2.5` boundary (`BETWEEN` vs. `<`).

| Band | Count | Percentage |
|---|---|---|
| `medium` | 2,574,777 | 41.8% |
| `high` | 2,489,182 | 40.4% |
| `low` | 563,231 | 9.1% |
| `invalid` | 529,099 | 8.6% |

**Difference:** Variant B reduces `low` from 17.8% to 9.1% (a drop of ~534k rows), because all `rate = 0` rows are moved to `invalid`. It also fixes a subtle boundary issue in Variant A: under `BETWEEN 0 AND 2.5`, the value `rate = 2.5` was being captured by the `low` clause (since `BETWEEN` is inclusive), incorrectly placing ~5,438 rows in `low`. Variant B places `rate = 2.5` correctly in `medium`.

**Suspicion (to be confirmed in Part C):** `rate = 0` exhibits anomalous behavior:
- **80.5%** of `rate = 0` rows carry the `recommended` label — a contradiction under the "zero stars = very dissatisfied" interpretation.
- The `rate` distribution has an unnatural spike at 0 (8.59%), higher than `rate = 1` (5.32%) and `rate = 2` (nearly absent). This pattern is more consistent with "missing star rating" than with a valid 0–5 satisfaction scale.

These observations **suggest** (but do not prove) that `rate = 0` is noise. The definitive test will be conducted in Part C: two `rating_band` models will be trained (Variant A with `rate = 0` → `low`, Variant B without) and compared on Macro F1, ordinal distance error, and confusion matrix. The final decision will be based on that comparison, not on the suspicion alone.

**Decision (provisional):** Variant B will be used for the initial `rating_band` training in Part C, with Variant A kept as a comparison baseline. The final choice will be validated empirically.

**Artifacts:**(reports/fig6_rating_band.png)

### A6. Vocabulary-Level Analysis (Optional Extension)

**Sampling note:** This analysis uses a random sample of **123,334 reviews** (out of 6.16M). The sample is drawn from the full dataset; global statistics (A1–A5) remain computed on the full data.

**Most frequent words:** The most frequent tokens mix stopwords and sentiment-bearing words:

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

The frequent sentiment words (`خوب`, `عالی`, `راضی`) and product attributes (`کیفیت`, `قیمت`) confirm that reviews center on quality and price.

**Class-specific bigrams:**

| Class | Representative bigrams |
|---|---|
| `recommended` | «خیلی کیوته», «باصرفه», «خوبه توصیه», «کاملا مطابق», «راضی بودیم», «مناسب قیمت» |
| `not_recommended` | «پول دور», «اصلا نخرید», «آشغال», «افتضاح بود», «مزخرف», «درد نخور», «حیف پول» |
| `no_idea` | «خاصی ندیدم», «مطمئن نیستم», «تاثیر خاصی», «خیلی معمولی», «مناسب نبود» |
| `<MISSING>` | «موجودش», «لطفا موجود کنید», «نخریدم», «بیرون خریدم», «نگرفتم», «چراغ قوه» |

**Key finding — why `<MISSING>` rows lack labels:** The bigrams for `<MISSING>` reveal two distinct user groups:
1. **Users who have not bought the product** («نخریدم» = I didn't buy, «بیرون خریدم» = I bought elsewhere, «نگرفتم» = I didn't get it) — they cannot recommend or not recommend a product they haven't used.
2. **Users who only request restocking** («موجودش» = make it available, «لطفا موجود کنید» = please restock) — their comment is about availability, not the product.

This **confirms the A2 finding** that missing labels are structural, not random: unlabeled users genuinely cannot provide a recommendation.

**Stopword handling:** Stopwords (`به`, `از`, `که`, `من`, `برای`) are frequent but are **not removed** at this stage. As the project requires, stopword removal will be treated as an **experiment** in Part B, not as a fixed preprocessing rule — because negations (`نه`, `نیست`) and discourse markers (`ولی`) carry sentiment information.

**What I will do:** In Part B, I will test two normalization variants (with and without stopword removal) and compare their impact on the TF-IDF baseline in Part C.

**Artifacts:** `reports/A6_frequent_words.csv`, `reports/A6_class_bigrams.csv`

**Figure 5 — Median body length by label.** The median body length increases monotonically from satisfied to dissatisfied users:

| Label | Median length (chars) |
|---|---|
| `<MISSING>` | 25 |
| `recommended` | 27 |
| `no_idea` | 38 |
| `not_recommended` | **48** |

Dissatisfied users write reviews that are ~1.8× longer than satisfied users (48 vs. 27 chars). This is a well-known psychological pattern: dissatisfied users need to justify their negative experience, while satisfied users often leave a brief "خوبه / عالیه". The `<MISSING>` group is the shortest — consistent with the A6 finding that unlabeled users are either restock-requesters or non-buyers who have little to say about the product.

### A7. Summary and Decision Mapping

This section summarizes the five key findings from Part A and maps each to a modeling decision.

| # | Finding (with evidence) | Modeling Decision |
|---|---|---|
| 1 | **Severe label imbalance:** `recommended` = 67.6%, `not_recommended` = 8.2% (A3-a). | Use **class weighting** and **Macro F1** instead of accuracy. |
| 2 | **Short reviews dominate:** median = 106 chars (~22 words); 97% of reviews < 297 chars (A4). | Set Transformer `max_length = 128` tokens. |
| 3 | **High repetition:** 197,746 duplicate bodies cover 36% of all rows; 75k bodies carry conflicting labels (A5). | **Flag** duplicates and conflicts (`is_duplicate_body`, `is_label_conflict`), do **not** remove them. |
| 4 | **`rate = 0` is anomalous:** 80.5% of `rate = 0` rows are labeled `recommended`; the distribution has an unnatural spike at 0 (A3-c). | Build two `rating_band` variants (A: standard, B: exclude `rate = 0`) and compare them in Part C. |
| 5 | **Missing labels are structural:** 100% of non-buyers lack `recommendation_status`; missingness is tied to `is_buyer` and `rate` (A2). | Keep unlabeled rows in the natural unlabeled pool; do **not** discard them. |

**Data provenance:**
- Figures 1–5 and all tables (A1–A5) are computed on the **full dataset** (6,156,289 rows).
- A6 (vocabulary analysis) uses a **random sample of 123,334 reviews** (seed = 42).
- All global counts (row counts, class distributions, missing rates) are computed on the full data.

**Artifacts:**
- `reports/fig1_comments_per_product.png`
- `reports/fig2_labels_and_rate.png`
- `reports/fig3_body_length.png`
- `reports/fig4_label_share_by_rate.png`
- `reports/fig5_label_share_by_rate.png`
- `reports/fig6_label_share_by_rate.png`
- `reports/fig7_length_by_label.png`
- `reports/missing_by_column.csv`
- `reports/top_repeated_bodies.csv`
- `reports/A4_quantiles.csv`
- `reports/A4_short_long.csv`
- `reports/A4_length_by_label.csv`
- `reports/A6_frequent_words.csv`
- `reports/A6_class_bigrams.csv`

---

## Part B — Cleaning, Splitting, and Label Protocol

This section describes the deterministic pipeline that transforms the raw dataset into the standard train/validation/test splits, including Persian text normalization, label mapping, rating-band derivation, product-grouped splitting, and the semi-supervised masking protocol. The entire pipeline is reproducible via `src/prepare_data.py`.

### B1. Persian Text Normalization

The normalization function (`src/text_norm.py`) applies the following transformations to the `body` column:

| Transformation | Examples |
|---|---|
| Arabic → Persian letters | `ي → ی`, `ك → ک`, `ى → ی` |
| Persian/Arabic digits → ASCII | `۰۱۲۳ → 0123`, `٠١٢٣ → 0123` |
| Remove diacritics (harakat) | `U+064B` to `U+0652` |
| Remove HTML tags | `<div>`, `<br>` |
| Remove URLs | `https://...`, `www....` |
| Collapse repeated letters | `خووووب → خووب` (3+ → 2) |
| Zero-width space (`U+200B`) → regular space | |
| Remove zero-width joiner / LRM / RLM / BOM | `U+200D`, `U+200E/200F`, `U+FEFF` |
| **Keep** ZWNJ (`U+200C`) | Preserves Persian half-spaces: `می‌روم`, `کتاب‌ها` |
| Keep emojis | Sentiment-bearing emojis are preserved |
| No stopword / negation removal | `نه`, `نیست`, `ولی` are kept |

After normalization, **215 rows** became empty (only URLs, HTML tags, or whitespace) and were dropped. The final normalized dataset contains **6,152,845 rows**.

### B2. Label Mapping and Quality Gate

The raw `recommendation_status` column was audited for unknown values:

| Raw value | Mapped label |
|---|---|
| `recommended` | `recommended` |
| `not_recommended` | `not_recommended` |
| `no_idea` | `no_idea` |
| `"nan"` (string) | `NULL` (missing label) |

The quality gate excluded:
- **3,229 duplicate IDs** (kept the first by `hash(body), product_id`)
- **1 row with `rate = 2500`** (outside the valid 0–5 range)
- **215 rows empty after normalization**

No rows had missing `id`, `product_id`, or `body` (before normalization).

### B3. Rating-Band Derivation (Variant B)

The `rating_band` target was derived using **Variant B** (see A3-d):

| Band | Threshold | Count (train) | Percentage (train) |
|---|---|---|---|
| `low` | `0 < rate < 2.5` | 393,941 | 9.2% |
| `medium` | `2.5 ≤ rate < 4.0` | 740,945 | 17.2% |
| `high` | `4.0 ≤ rate ≤ 5.0` | 2,801,794 | 65.0% |
| `NULL` | `rate = 0`, `rate > 5`, or missing | 370,310 | 8.6% |

The ordinal encoding is `low=0`, `medium=1`, `high=2` (stored in `split_manifest.json`). `rate = 0` is treated as `NULL` (likely a missing star rating, not a valid zero-star rating). This choice will be validated in Part C by comparing Variant A (standard) and Variant B.

### B4. Product-Grouped Split

A **product-grouped split** was used to prevent leakage. Products are sorted by `(-row_count, sha256(SPLIT_SEED | product_id))` and assigned greedily to minimize the normalized deviation from the 70/15/15 target across rows, 3 recommendation classes, and 3 rating bands.

| Split | Products | Rows | Row % | Recommended | Not Recommended | No Idea | Label Missing |
|---|---|---|---|---|---|---|---|
| **train** | 294,063 | 4,306,990 | 69.97% | 2,912,438 (67.6%) | 353,151 (8.2%) | 415,585 (9.6%) | 625,816 (14.5%) |
| **validation** | 18,762 | 922,927 | 14.99% | 624,094 (67.6%) | 75,675 (8.2%) | 89,054 (9.6%) | 134,104 (14.5%) |
| **test** | 18,767 | 922,928 | 15.03% | 624,094 (67.6%) | 75,675 (8.2%) | 89,054 (9.6%) | 134,105 (14.5%) |

The class distribution is nearly identical across splits, confirming that the greedy algorithm preserved representativeness.

### B5. Semi-Supervised Masking Protocol

The masking follows the project specification exactly:
- `SPLIT_SEED = 2026`, `MASK_SEED = "digikala-semi-supervised-v1"`, `MASK_RATE = 0.20`
- For each recommendation class in the training split, sort rows by `(stable_hash(id), id)` and mask `floor(0.20 × class_count)` rows.
- Masked rows are moved to the artificial unlabeled pool. Validation and test rows are never masked.

| Class | Labeled train rows | Masked | Kept supervised |
|---|---|---|---|
| `recommended` | 2,912,438 | 582,487 | 2,329,951 |
| `no_idea` | 415,585 | 83,117 | 332,468 |
| `not_recommended` | 353,151 | 70,630 | 282,521 |
| **Total** | 3,681,174 | **736,234** | 2,944,940 |

The total masked count (**736,234**) matches the size of `train_pseudo_unlabeled.csv` exactly (verified by assertion in the pipeline).

### B6. Output Files

| File | Rows | Description |
|---|---|---|
| `train_supervised.csv` | 3,555,676 | Train rows not masked, with at least one valid target (`recommendation_status` and/or `rating_band`) |
| `train_pseudo_unlabeled.csv` | 736,234 | Masked train rows with only `id` and `body` (labels removed) |
| `natural_unlabeled.csv` | 625,816 | Train rows with missing `recommendation_status` (id, body only) |
| `validation.csv` | 917,793 | Validation rows with at least one valid target (never masked) |
| `test.csv` | 917,785 | Test rows with at least one valid target (never masked) |

The artificial unlabeled file contains only `id` and `body` — `recommendation_status`, `rating_band`, and `rate` were removed (as required). An assertion verifies that `train_pseudo_unlabeled.csv` has exactly these two columns.

### B7. Split Manifest

The complete manifest is saved to both `data/processed/split_manifest.json` and `reports/split_manifest.json`, and records:
- seeds (`SPLIT_SEED`, `MASK_SEED`, `MASK_RATE`);
- split strategy and target fractions;
- achieved row fractions and products per split;
- rating-band thresholds and encoding (Variant B);
- label mapping;
- normalization rules;
- exclusions and duplicate policy;
- per-split class and band statistics;
- per-class masking counts;
- output file row counts.

**Artifacts:**
- `data/processed/normalized.parquet`
- `data/processed/train_supervised.csv`
- `data/processed/train_pseudo_unlabeled.csv`
- `data/processed/natural_unlabeled.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/split_manifest.json`
- `reports/split_manifest.json`

---

## Part C — Modeling

### C1. TF-IDF Feature Extraction

The first step of modeling is converting the normalized Persian text into numerical features. A **TF-IDF vectorizer** was built with the following configuration:

| Parameter | Value | Rationale |
|---|---|---|
| `ngram_range` | `(1, 2)` | Captures both single words and bigrams (e.g., «خوب بود», «توصیه می‌کنم») |
| `min_df` | `5` | Removes rare tokens (typos, proper nouns) that the model cannot learn from |
| `max_features` | `400,000` | Caps vocabulary size to control memory |
| `sublinear_tf` | `True` | Applies logarithmic TF scaling (standard for TF-IDF) |
| `token_pattern` | `r"(?u)[\w\u200c]{2,}"` | Keeps ZWNJ (`\u200c`) as part of words, so «می‌روم» stays one token |
| `dtype` | `float32` | Reduces memory usage |

**Fit strategy:** To keep the pipeline fast, the vectorizer was fit on a **random sample of 500,000 training reviews** (with `random_state=SPLIT_SEED=2026`). The fitted vocabulary was then applied to the full train, validation, and test sets. Fitting on the full 3.5M training set would be slower and would not meaningfully change the vocabulary, since the sample already captures the frequent tokens.

**Output:**

| Metric | Value |
|---|---|
| Vocabulary size | **115,840** tokens |
| `X_train` shape | **(3,555,676, 115,840)** |
| `X_train` non-zero entries (`nnz`) | **53,136,203** |
| Density | **0.013%** (highly sparse) |
| Fit time | 45 s |
| Total transform time | 165 s (~2.7 min) |

The resulting matrix is **highly sparse**: each review contains on average ~15 non-zero tokens (53M ÷ 3.5M), consistent with the short-review finding from A4 (median 22 words). A dense representation would require ~1.6 TB; storing the matrix in sparse CSR format keeps it manageable.

**Caching:** The vectorizer (`tfidf_vectorizer.joblib`) and the three sparse matrices (`X_train.npz`, `X_val.npz`, `X_test.npz`) are cached under `artifacts/tfidf/`. If these artifacts exist, the pipeline reloads them instead of recomputing, ensuring reproducibility and fast iteration.

**Artifacts:**
- `artifacts/tfidf/tfidf_vectorizer.joblib`
- `artifacts/tfidf/X_train.npz`
- `artifacts/tfidf/X_val.npz`
- `artifacts/tfidf/X_test.npz`

### C2. TF-IDF Baseline: Hyperparameter Tuning

Two baselines were trained and compared to quantify the effect of increasing `max_iter` and refining the `alpha` search grid.

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

**What I observed:** Increasing `max_iter` from 15 to 50 and using a finer alpha grid (1.5e-6 to 5e-6) produced **negligible improvements** (+0.0001 and +0.0018). The `ConvergenceWarning` observed during the original run was benign — SGD had effectively converged even at `max_iter=15`. The alpha curve is nearly flat between `3e-6` and `5e-6` for both tasks.

**What it means:** The TF-IDF + SGD pipeline has reached its **ceiling** at ~0.68 (recommendation) and ~0.65 (rating_band). The bottleneck is not the optimizer settings but the **TF-IDF representation itself**, which cannot capture word order, negation, or context — limitations that are particularly harmful given the short reviews (34% under 20 chars) and the label noise (75k conflicting labels, 36% duplicate bodies) identified in Part A.

**What I will do:** Use the **improved baseline** for comparison with the Transformer model, since it represents the strongest classical baseline. The original baseline is reported for transparency and to document the effect of the hyperparameter change.

### C3. TF-IDF Baseline — Final Evaluation on Test

The final TF-IDF models were trained on the full training set with the tuned `alpha` and evaluated on validation and test.

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

**What I observed:** Test results are **slightly better than validation** (Macro F1: 0.699 vs. 0.683 for recommendation; 0.658 vs. 0.647 for rating_band), indicating the model is **not overfitting** and benefits from full-train training. The dominant class `recommended` achieves F1 = 0.930, while `no_idea` is the weakest (F1 = 0.450). For `rating_band`, `high` is strong (F1 = 0.881) but `medium` is weak (F1 = 0.415). The ordinal distance error is low (0.245), meaning errors are mostly between adjacent bands, not distant ones.

**What it means:** The TF-IDF baseline performs well on the dominant classes but struggles on ambiguous or minority classes:
- `no_idea` is inherently ambiguous (e.g., «معمولی», «مطمئن نیستم») and overlaps with both `recommended` and `not_recommended`.
- `medium` has artificial boundaries (2.5–4.0) and is affected by the 119k conflicting-rate rows identified in A5.
- The gap between Accuracy (0.853) and Macro F1 (0.699) is expected under class imbalance: the model is excellent on the majority class but weaker on the two minority classes.

**What I will do:** Use these results as the classical baseline for comparison with the Transformer model. The Transformer is expected to improve on the minority classes by capturing word order, negation, and context — which TF-IDF cannot. The confusion matrices and top features will inform the error analysis in the final report.

**Artifacts:**
- `reports/preds_tfidf_recommendation_validation.parquet`
- `reports/baseline_metrics.json`
- `reports/fig_baseline_confusion.png`
- `artifacts/tfidf/logreg_recommendation.joblib`
- `artifacts/tfidf/logreg_rating_band.joblib`

**Confusion matrices (row-normalized) — Figure C3:**

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

**What I observed:** Both tasks exhibit the same failure pattern: the model **defaults to the dominant class when uncertain**.
- **`no_idea`:** 33% is misclassified as `recommended` (the majority class); only 46% is correctly predicted.
- **`medium`:** 47% is misclassified as `high` (the majority band); only 36% is correctly predicted.
- In contrast, the dominant classes are well captured: `recommended` (92%) and `high` (90%).

**What it means:** The confusion matrices reveal the **ceiling of the TF-IDF representation**: it cannot separate ambiguous minority classes (`no_idea`, `medium`) from the dominant ones, because it lacks word-order, negation, and contextual information. `class_weight="balanced"` reweights the loss, but it cannot fix the representation itself. This is the strongest empirical evidence that a contextual model (Transformer) is needed.

**What I will do:** Use these confusion matrices as the baseline for the error analysis in the final report. The `no_idea` → `recommended` and `medium` → `high` confusions will be specifically inspected to see whether the Transformer reduces them.

**Artifacts:** `reports/fig_baseline_confusion.png`

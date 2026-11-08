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
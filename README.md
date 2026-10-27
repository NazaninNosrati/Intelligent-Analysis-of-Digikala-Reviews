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

## 🗂️ Repository Structure

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
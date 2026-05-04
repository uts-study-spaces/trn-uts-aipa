# Group Run Checklist

Use this checklist when a group member pulls the repo and wants to run the notebooks, training pipeline, or Streamlit demo.

## 1. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Optional, if `uv` is installed:

```powershell
uv sync
```

## 2. Dataset

Download the Kaggle dataset:

https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets

Place these compatible CSV files in `data/raw/`:

```text
data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv
data/raw/dataset-tickets-multi-lang-4-20k.csv
data/raw/dataset-tickets-multi-lang3-4k.csv
```

For notebook `01_eda.ipynb`, also place these audit-only files in `data/raw/`:

```text
data/raw/dataset-tickets-german_normalized.csv
data/raw/dataset-tickets-german_normalized_50_5_2.csv
```

The raw CSV files are ignored by Git because they are dataset files, not source code.

## 3. Notebook Order

Run the notebooks in this order:

```text
notebooks/01_eda.ipynb
notebooks/02_cleaning_feature_engineering.ipynb
notebooks/03_baseline_models.ipynb
notebooks/04_experiments_and_pipeline.ipynb
notebooks/05_final_evaluation_and_error_analysis.ipynb
notebooks/06_transformer_embedding_benchmark.ipynb  optional, requires extra dependencies
```

Notebook `04` regenerates model files and main results.
Notebook `06` is optional and supports the final report discussion about transformer-based semantic representations.

## 4. Command-Line Training

```powershell
.\.venv\Scripts\python.exe -m customer_support_ai.train
```

This saves:

```text
data/processed/model_ready_tickets.csv
models/category_model.pkl
models/priority_model.pkl
results/metrics_summary.csv
results/classification_reports.json
results/confusion_matrix_category.png
results/confusion_matrix_priority.png
```

## 5. Report Assets

```powershell
.\.venv\Scripts\python.exe -m customer_support_ai.report_assets
```

This creates presentation/report charts in `report_assets/` and extra evaluation tables in `results/`.

## 6. Streamlit Demo

```powershell
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

Use `docs/presentation_demo_examples.md` for stable demo inputs.

## 7. Optional Transformer Benchmark

Install the optional dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-transformer.txt
```

Run the sampled benchmark:

```powershell
.\.venv\Scripts\python.exe -m customer_support_ai.transformer_benchmark --sample-size 2500
```

This saves `results/transformer_embedding_benchmark.csv` and `results/transformer_embedding_benchmark.json`. Use it for report discussion, not for the live Streamlit demo.

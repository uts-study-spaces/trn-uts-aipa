# Customer Support AI Implementation

<<<<<<< HEAD
This repository contains the actual AI implementation for the Assessment 3 group project in Artificial Intelligence Principles and Applications at UTS. The system supports customer service teams by classifying support tickets, predicting priority, recommending a support team, summarising the issue, explaining key prediction terms, flagging escalation risk, and analysing small batches of tickets through the demo app.
=======
This repository contains the actual AI implementation for the Assessment 3 group project in Artificial Intelligence Principles and Applications at UTS. The system supports customer service teams by classifying support tickets, predicting priority, recommending a support team, summarising the issue, explaining key prediction terms, and flagging escalation risk.
>>>>>>> 1fd79b5 (Add MLP classifier, RL routing agents)

The Streamlit app is only the final user-facing interface. The main project work is the reproducible AI pipeline in `src/` and the staged analysis notebooks in `notebooks/`.

## Real-World Problem

Customer support teams receive high volumes of tickets across account access, billing, security, bugs, performance, feature requests, and integration issues. Manual triage can be slow, inconsistent, and risky when urgent or security-related tickets are missed. This project implements an AI decision-support workflow to assist first-pass ticket triage while keeping final responsibility with human support staff.

## AI Implementation

- NLP preprocessing for customer ticket text.
- Safe feature engineering using ticket text and pre-resolution metadata.
- Target leakage control by excluding post-resolution fields such as resolution notes and status.
- Train/validation/test split for fair model selection and final evaluation.
- Baseline and comparison models:
  - TF-IDF + Logistic Regression
  - TF-IDF + Linear SVM
  - TF-IDF + Naive Bayes
<<<<<<< HEAD
- Optional multilingual transformer-embedding benchmark for final report discussion.
=======
>>>>>>> 1fd79b5 (Add MLP classifier, RL routing agents)
- Separate supervised models for category prediction and priority prediction.
- Rule-based reasoning for routing and escalation.
- Template-based summarisation with privacy masking.
- Explainability using influential TF-IDF terms.
<<<<<<< HEAD
- Streamlit interface with overview evidence, single-ticket prediction, batch upload, and downloadable batch predictions.
- Report-ready metrics and confusion matrices.
- LLM agent (AI Assistant tab) that uses Gemini to orchestrate the trained ML models as tools to triage tickets, draft suggested replies, and summarise batch results.
=======
- Report-ready metrics and confusion matrices.
>>>>>>> 1fd79b5 (Add MLP classifier, RL routing agents)

## Repository Structure

```text
<<<<<<< HEAD
app/                         Streamlit interface for overview, single-ticket demo, and batch demo
data/raw/                    Full Kaggle dataset location, not committed
data/processed/              Generated model-ready dataset, not committed
docs/                        Project plan, run checklist, demo examples, and assignment notes
models/                      Trained model files, not committed
notebooks/                   Clean staged notebooks for final workflow
reports/                     Status report and executed notebook copies
=======
app/                         Streamlit interface for final demonstration
data/raw/                    Full Kaggle dataset location, not committed
data/processed/              Generated model-ready dataset, not committed
docs/                        Project plan and assignment notes
models/                      Trained model files, not committed
notebook/                    Existing exploratory notebook from the group
notebooks/                   Clean staged notebooks for final workflow
>>>>>>> 1fd79b5 (Add MLP classifier, RL routing agents)
report_assets/               Generated charts for report and presentation
results/                     Metrics, reports, and charts for report use
src/customer_support_ai/     Reusable implementation code
```

## Notebook Workflow

```text
01_eda.ipynb
  Project landing, dataset loading, EDA, missing values, duplicates, label balance, and text patterns.

02_cleaning_feature_engineering.ipynb
  Data cleaning, NLP preprocessing, metadata handling, leakage control, and model-ready features.

03_baseline_models.ipynb
  Baseline model comparison for category and priority prediction.

04_experiments_and_pipeline.ipynb
  Main training pipeline, model selection, and saved report outputs.

05_final_evaluation_and_error_analysis.ipynb
  Test results, example predictions, failure cases, limitations, and responsible AI reflection.
<<<<<<< HEAD

06_transformer_embedding_benchmark.ipynb
  Optional sampled comparison of TF-IDF with multilingual transformer embeddings for SLO4/future-work discussion.
=======
>>>>>>> 1fd79b5 (Add MLP classifier, RL routing agents)
```

## Setup

Create a local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

If your machine has uv installed, this also works:

```powershell
uv sync
```

<<<<<<< HEAD
Optional transformer benchmark dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-transformer.txt
```

=======
>>>>>>> 1fd79b5 (Add MLP classifier, RL routing agents)
## Dataset

Download the Kaggle Multilingual Customer Support Tickets dataset:

https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets

Recommended method:

```powershell
.\.venv\Scripts\python.exe -m customer_support_ai.download_data
```

This uses `kagglehub` to download the Kaggle dataset and copies the three compatible CSV files here:

```text
data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv
data/raw/dataset-tickets-multi-lang-4-20k.csv
data/raw/dataset-tickets-multi-lang3-4k.csv
```

Manual method:

Download the Kaggle CSV files and copy these three compatible files into the repo:

```text
data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv
data/raw/dataset-tickets-multi-lang-4-20k.csv
data/raw/dataset-tickets-multi-lang3-4k.csv
```

Notebook `01_eda.ipynb` also audits the two German-normalized CSV files from Kaggle:

```text
data/raw/dataset-tickets-german_normalized.csv
data/raw/dataset-tickets-german_normalized_50_5_2.csv
```

These two files are not merged into the main supervised model because their queue/category taxonomies are different from the main ticket-routing labels.

If Kaggle asks for authentication, sign in to Kaggle and create an API token from your Kaggle account settings. Save `kaggle.json` in the standard Kaggle credentials location for your machine, then rerun the command. In Kaggle notebooks, authentication is normally handled by Kaggle automatically.

The raw CSV files are intentionally ignored by Git. Each group member should download the same Kaggle files and place them at the same repo-relative paths. The modelling pipeline uses `subject` and `body` as the ticket text, predicts `queue` and `priority`, and excludes `answer` because it is only available after support staff respond.

## Train and Evaluate Models

Final training with the compatible merged Kaggle dataset:

```powershell
.\.venv\Scripts\python.exe -m customer_support_ai.download_data
<<<<<<< HEAD
.\.venv\Scripts\python.exe -m customer_support_ai.train --vectorizer tfidf
```

### Switchable Text Representations

You can now compare different NLP embedding methods by using the `--vectorizer` flag:

- `tfidf` (Default): Traditional sparse representation.
- `word2vec`: Dense word embeddings averaged per ticket.
- `top2vec`: Document embeddings using the Top2Vec library.

Example:
```powershell
.\.venv\Scripts\python.exe -m customer_support_ai.train --vectorizer word2vec
=======
.\.venv\Scripts\python.exe -m customer_support_ai.train
>>>>>>> 1fd79b5 (Add MLP classifier, RL routing agents)
```

Training saves:

- `data/processed/model_ready_tickets.csv`
- `models/category_model.pkl`
- `models/priority_model.pkl`
- `results/dataset_profile.json`
- `results/metrics_summary.csv`
- `results/classification_reports.json`
- `results/confusion_matrix_category.png`
- `results/confusion_matrix_priority.png`

## Generate Report and Presentation Assets

After training the models:

```powershell
.\.venv\Scripts\python.exe -m customer_support_ai.report_assets
```

This creates dataset, model comparison, per-class, per-language, and workflow charts in `report_assets/`. It also saves `results/per_class_f1.csv` and `results/per_language_metrics.csv` for report tables.

<<<<<<< HEAD
## Optional Transformer-Embedding Benchmark

The main app still uses TF-IDF + Linear SVM because it is lightweight, reproducible, and explainable. For the final report, the project also includes an optional sampled benchmark that compares sparse TF-IDF features with dense multilingual transformer embeddings.

Install the optional dependency and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-transformer.txt
.\.venv\Scripts\python.exe -m customer_support_ai.transformer_benchmark --sample-size 2500
```

This saves:

- `results/transformer_embedding_benchmark.csv`
- `results/transformer_embedding_benchmark.json`

The benchmark is not used for live ticket prediction. If the benchmark output files exist, Streamlit displays them in the Overview tab alongside the main TF-IDF model results so they can support report and presentation discussion.

=======
>>>>>>> 1fd79b5 (Add MLP classifier, RL routing agents)
## Run the Final Interface

After training the models:

```powershell
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

<<<<<<< HEAD
The interface has two tabs:

- `Overview`: shows dataset size, compatible CSV count, duplicates removed, best test macro F1, workflow, language/priority charts, final metrics, per-language and per-class results, and the optional transformer benchmark table/chart when available.
- `Try Solution`: supports single-ticket analysis and batch upload. Single-ticket mode returns the predicted category, predicted priority, recommended team, summary, escalation flag, and explanation terms. Batch mode accepts CSV, XLSX, or XLS files with a ticket text column and analyses up to 200 rows, then offers a downloadable predictions CSV.

Stable presentation examples are available in `docs/presentation_demo_examples.md`.

## Using the AI Assistant

Navigate to **Try Solution → AI Assistant**. Enter a Gemini API key when prompted - a working key is provided in the Canvas report submission under Section 7 - GitHub Repository to avoid key misuse.

Type or paste a support ticket into the chat input and press Enter to triage it. To process a batch, click the **+** button in the chat input to attach a CSV or Excel file.
=======
The interface lets a user paste a support ticket and returns the predicted category, predicted priority, recommended team, summary, escalation flag, and explanation terms. Stable presentation examples are available in `docs/presentation_demo_examples.md`.
>>>>>>> 1fd79b5 (Add MLP classifier, RL routing agents)

## Responsible AI Notes

This system is decision support, not full automation. It avoids using resolution notes as prediction input because that would leak information unavailable at ticket creation time. Summaries mask obvious emails and phone numbers. Escalation rules prioritise human review for security, fraud, urgent, and critical cases. The final report should discuss misclassification risk, historical label bias, privacy, over-reliance on automation, scalability, and the need for human review.

## Current Status

- Actual implementation pipeline is in place and configured for the compatible merged Kaggle CSV files.
- Staged notebooks 01 to 05 run successfully with the merged local dataset.
<<<<<<< HEAD
- Optional notebook 06 runs a sampled multilingual transformer-embedding benchmark for comparison and future-work discussion.
- Current strongest model is Linear SVM for both queue and priority prediction.
- Current modelling dataset has 44,275 usable rows after auditing five CSV files and deduplicating the three compatible files.
- Report-ready metrics, classification reports, confusion matrices, extra evaluation tables, slide charts, and example success/failure cases have been regenerated.
- Streamlit now includes an Overview evidence dashboard, a single-ticket demo, a batch upload workflow, dark report tables, Altair charts, and optional transformer benchmark display.
=======
- Current strongest model is Linear SVM for both queue and priority prediction.
- Current modelling dataset has 44,275 usable rows after auditing five CSV files and deduplicating the three compatible files.
- Report-ready metrics, classification reports, confusion matrices, extra evaluation tables, slide charts, and example success/failure cases have been regenerated.
>>>>>>> 1fd79b5 (Add MLP classifier, RL routing agents)
- Group member names, student IDs, screenshots, and final contribution statements still need to be added before submission.

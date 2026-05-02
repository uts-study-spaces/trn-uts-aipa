# Customer Support AI Implementation Plan

## Aim

Build an explainable AI decision-support implementation for customer service teams. The system predicts ticket category and priority, recommends a team, creates a short summary, explains model decisions, and flags escalation risk.

## HD Alignment

- Multiple AI paradigms: supervised NLP classification, rule-based reasoning, knowledge representation through routing rules, and template-based summarisation.
- Theoretical justification: compare TF-IDF Logistic Regression, Linear SVM, and Naive Bayes as explainable text classification methods.
- Empirical evaluation: use train/validation/test splitting and report accuracy, macro precision, macro recall, macro F1, weighted F1, and confusion matrices for category and priority.
- Additional analysis: include per-class F1 and per-language performance to support critical reflection on multilingual behaviour.
- Responsible AI: avoid target leakage, anonymise obvious personal data in summaries, use explanation terms, and keep humans responsible for final decisions.

## Dataset

Main dataset: Kaggle Multilingual Customer Support Tickets dataset.

Kaggle provides five CSV files. Notebook `01_eda.ipynb` audits all five files, then the project merges only the three files with compatible `queue` and `priority` labels.

Compatible modelling CSV files:

```text
aa_dataset-tickets-multi-lang-5-2-50-version.csv
dataset-tickets-multi-lang-4-20k.csv
dataset-tickets-multi-lang3-4k.csv
```

Audit-only CSV files:

```text
dataset-tickets-german_normalized.csv
dataset-tickets-german_normalized_50_5_2.csv
```

The audit-only files are not merged into the main queue classifier because their queue/category labels use different taxonomies. The compatible merge produces 44,275 usable modelling rows after deduplication and basic filtering, with English, German, Spanish, French, and Portuguese examples. The key fields for this project are:

- `subject` and `body`: initial customer ticket text.
- `queue`: support queue/category to predict.
- `priority`: ticket priority to predict.
- `type`, `language`, and tags: safe metadata available at ticket creation time.
- `answer`: excluded from prediction because it is written after support staff respond.

Place the compatible downloaded CSV files at:

```text
data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv
data/raw/dataset-tickets-multi-lang-4-20k.csv
data/raw/dataset-tickets-multi-lang3-4k.csv
```

Recommended loading method:

```powershell
python -m customer_support_ai.download_data
```

This uses `kagglehub`, which downloads the dataset from Kaggle and stores the compatible CSV files in `data/raw/`. The raw CSV files are kept out of Git, so each group member should place the same Kaggle files under `data/raw/`.

## Technical Workflow

1. Load and inspect all five Kaggle CSV files.
2. Compare schema, languages, queue labels, priority labels, and compatibility.
3. Merge the three compatible multilingual CSV files.
4. Deduplicate tickets using subject plus body.
5. Normalise column names and remove unusable rows.
6. Combine subject and body into one `ticket_text` field.
7. Clean ticket text while preserving important urgency, billing, login, and security words.
8. Build a safe model input from initial ticket text and non-leaking metadata.
9. Train three models for queue prediction and three models for priority prediction.
10. Split data into training, validation, and testing sets.
11. Select the best model for each task using validation macro F1-score.
12. Retrain the selected model on training plus validation data.
13. Evaluate final performance on the held-out test set.
14. Save metrics, classification reports, and confusion matrices.
15. Generate report and presentation charts from the saved results.
16. Analyse per-class and per-language performance.
17. Use rule-based reasoning for routing and escalation.
18. Use TF-IDF feature weights for explanation terms.
19. Present the final trained system through a Streamlit interface.

## Notebooks

- `01_eda.ipynb`: project landing, dataset loading, and EDA.
- `02_cleaning_feature_engineering.ipynb`: NLP cleaning, feature engineering, and leakage control.
- `03_baseline_models.ipynb`: baseline model comparison.
- `04_experiments_and_pipeline.ipynb`: reusable training pipeline and model selection.
- `05_final_evaluation_and_error_analysis.ipynb`: final metrics, examples, failure cases, and reflection.

## Report and Presentation Assets

Run this command after model training:

```powershell
python -m customer_support_ai.report_assets
```

Generated assets include dataset distribution charts, model comparison charts, per-class F1 charts, per-language macro F1 chart, and a workflow diagram. These outputs support the Stage 1 presentation and Stage 2 report evidence.

## Group Work Split

- Member 1: dataset collection, EDA, and data quality notes.
- Member 2: preprocessing and leakage control.
- Member 3: category model training and comparison.
- Member 4: priority model training and comparison.
- Member 5: routing, escalation, summarisation, and explainability.
- Member 6: Streamlit demo, README, report assets, and integration testing.

Update the final names and student IDs before report submission.

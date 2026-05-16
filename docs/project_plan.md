# Customer Support AI Implementation Plan

## Aim

Build an explainable AI decision-support implementation for customer service teams. The system predicts ticket category and priority, recommends a team, creates a short summary, explains model decisions, flags escalation risk, and supports small batch triage through the demo app.

## HD Alignment

- Multiple AI paradigms: supervised NLP classification, TF-IDF structural text representation, a probabilistic Naive Bayes baseline, rule-based reasoning, MLP neural-network experimentation, reinforcement-learning routing exploration, optional transformer-embedding benchmarking, and optional Gemini-based LLM assistance.
- Theoretical justification: compare TF-IDF Logistic Regression, Linear SVM, and Naive Bayes as explainable text classification methods.
- Empirical evaluation: use train/validation/test splitting and report accuracy, macro precision, macro recall, macro F1, weighted F1, and confusion matrices for category and priority.
- Additional analysis: include per-class F1 and per-language performance to support critical reflection on multilingual behaviour.
- Responsible AI: avoid target leakage, anonymise obvious personal data in summaries, use explanation terms, and keep humans responsible for final decisions.

## Dataset

Main dataset: Kaggle **Multilingual Customer Support Tickets** dataset by Tobias Bueck.

Dataset link: https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets

In simple terms, the dataset is a collection of example customer support requests. Each row describes one ticket, including the customer's subject/body text, language, ticket type, queue/category, priority, tags, and an agent answer. The project uses the ticket text and safe pre-resolution metadata to predict the support queue and urgency. The `answer` field is excluded because it is written after the ticket is handled and would leak future information.

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
17. Test an MLP priority classifier as a multi-layer neural-network alternative.
18. Optionally compare TF-IDF features with multilingual transformer embeddings on a sampled benchmark for final report discussion.
19. Explore reinforcement-learning routing logic as a future operational extension.
20. Use rule-based reasoning for routing and escalation.
21. Use TF-IDF feature weights for explanation terms.
22. Present the final trained system through a Streamlit interface with Overview evidence, single-ticket analysis, batch upload, AI Assistant support, and downloadable batch predictions.

## Notebooks

- `01_eda.ipynb`: project landing, dataset loading, and EDA.
- `02_cleaning_feature_engineering.ipynb`: NLP cleaning, feature engineering, and leakage control.
- `03_baseline_models.ipynb`: baseline model comparison.
- `04_experiments_and_pipeline.ipynb`: reusable training pipeline and model selection.
- `05_final_evaluation_and_error_analysis.ipynb`: final metrics, examples, failure cases, and reflection.
- `06_transformer_embedding_benchmark.ipynb`: optional sampled benchmark using multilingual transformer embeddings plus a simple supervised classifier.

## Report and Presentation Assets

Run this command after model training:

```powershell
python -m customer_support_ai.report_assets
```

Generated assets include dataset distribution charts, model comparison charts, per-class F1 charts, per-language macro F1 chart, and a workflow diagram. These outputs support the Stage 1 presentation and Stage 2 report evidence.

The optional transformer benchmark saves `results/transformer_embedding_benchmark.csv` and `results/transformer_embedding_benchmark.json`. The benchmark is deliberately separate from the production-style app prediction path because it requires heavier dependencies and is run on a sample. When those result files exist, the Streamlit Overview tab displays them as comparison evidence alongside the main TF-IDF model metrics.

## Streamlit Demo

The demo app is in `app/streamlit_app.py`. It has:

- `Overview`: evidence dashboard with dataset profile, workflow, language and priority charts, final metrics, transformer benchmark display, per-language metrics, and per-class F1 tables.
- `Try Solution`: single-ticket prediction and a batch upload flow for CSV/XLSX/XLS files. Batch mode analyses up to 200 rows and exports a predictions CSV.

The live predictions use the trained TF-IDF Linear SVM models saved in `models/`. MLP, Word2Vec, Top2Vec, transformer embeddings, Gemini assistance, and reinforcement-learning routing are shown or discussed as comparison/extension evidence only; they do not replace the deployed prediction path.

## Group Work Split

- Member 1: dataset collection, EDA, and data quality notes.
- Member 2: preprocessing and leakage control.
- Member 3: category model training and comparison.
- Member 4: priority model training and comparison.
- Member 5: routing, escalation, summarisation, and explainability.
- Member 6: Streamlit demo, batch upload workflow, README, report assets, and integration testing.

Final report contributions should list each member's specific and verifiable work using repository history, notebooks, documents, and presentation artefacts.

# Reproducibility and Release Checklist

This checklist was added to make the repository easier to verify before final
submission.

## Environment

- Create a fresh virtual environment.
- Install `requirements.txt`.
- Install the project in editable mode with `pip install -e .`.
- Install `requirements-transformer.txt` only when the optional transformer
  benchmark is required.

## Data

- Download the Kaggle Multilingual Customer Support Tickets dataset.
- Place the three compatible modelling CSV files in `data/raw/`.
- Keep raw data out of Git.
- Confirm that the post-response `answer` field is not used for prediction.

## Training

- Run `python -m customer_support_ai.train --vectorizer tfidf`.
- Run `python -m customer_support_ai.report_assets`.
- Optionally run `python -m customer_support_ai.ensemble` for confidence-aware
  ensemble evidence.
- Optionally run `python -m customer_support_ai.transformer_benchmark --sample-size 2500`.

## Validation

- Run `python -m customer_support_ai.repository_health`.
- Run `python -m pytest`.
- Confirm that `results/metrics_summary.csv` contains no Git conflict markers.
- Confirm that result files and figures are present before report compilation.

## App

- Run `streamlit run app/streamlit_app.py`.
- Check the Overview tab for metrics and workflow evidence.
- Check the Try Solution tab for single-ticket analysis.
- Check the batch upload workflow with `docs/demo_upload_single.csv` or
  `docs/demo_upload_multiple.csv`.
- Confirm that confidence and human-review outputs are displayed.

## Submission

- Confirm that the report contains the final GitHub link.
- Confirm that individual contributions are listed with specific evidence.
- Confirm that the PowerPoint filename follows `AT3_Slide_Group_xxxx.pptx`.
- Confirm that the report filename follows `AT3_Report_Group_xxxx.pdf`.

# Model Card: Ticket Routing Intelligence

## Model purpose

Ticket Routing Intelligence is used as a decision-support system for first-pass
customer support triage. It predicts the likely support queue and priority of a
ticket, recommends a handling team, flags escalation risk, and provides
explanation terms for agent review.

## Intended use

The model is intended to assist support agents by reducing manual triage effort
and improving consistency. The output is advisory. Final routing, escalation,
and customer communication decisions remain with human staff.

## Not intended for

The model is not intended for fully autonomous customer service decisions,
disciplinary action against staff, fraud determination, legal decisions, or
high-risk use without human review and production monitoring.

## Data

The project uses the Kaggle Multilingual Customer Support Tickets dataset. Five
CSV files were audited and three compatible multilingual files were merged. The
modelling dataset contains 44,275 usable tickets after filtering and
deduplication. The post-response `answer` field is excluded to avoid target
leakage.

## Model family

The main deployed pipeline uses TF-IDF text representation with Linear SVM
classification for queue and priority. Additional comparison work includes
Logistic Regression, Naive Bayes, MLP, transformer embeddings, rule-based
routing, reinforcement-learning routing exploration, and a Gemini-backed AI
assistant.

## Amal contribution: confidence-aware ensemble

A confidence-aware ensemble extension was added as Amal's contribution to
support responsible deployment analysis. The ensemble combines Logistic
Regression, calibrated Linear SVM, and Naive Bayes with soft voting. Confidence,
coverage, human-review rate, and high-confidence accuracy are reported. This
extension is designed to identify cases where automated triage should be treated
cautiously.

## Evaluation

The main project reports accuracy, macro precision, macro recall, macro F1, and
weighted F1. Macro F1 is prioritised because support queues and priority levels
are not perfectly balanced. The confidence-aware extension adds coverage and
human-review metrics to connect model confidence with safe operational use.

## Ethical considerations

Potential harms include misrouting urgent tickets, exposing private customer
information, reproducing historical label bias, and over-reliance on automated
recommendations. Mitigations include leakage control, privacy masking,
explanation terms, escalation rules, confidence guidance, and human review.

## Limitations

Ticket categories can overlap, multilingual performance varies, and confidence
scores from non-calibrated models should be interpreted as decision-support
signals rather than exact probabilities. Production deployment would require
monitoring, access control, drift detection, stronger redaction, and evaluation
on organisation-specific data.

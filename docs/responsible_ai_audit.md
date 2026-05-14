# Responsible AI Audit

## Audit objective

This audit documents the responsible AI safeguards used in Ticket Routing
Intelligence and identifies controls that support safe human-in-the-loop
deployment. It was added to strengthen the project against the Assessment 3
criteria for ethics, critical reflection, trade-offs, and scalability.

## Key risks

| Risk | Potential impact | Existing control |
| --- | --- | --- |
| Misrouting | A ticket may be sent to the wrong team. | Queue prediction is treated as decision support, not final action. |
| Missed urgency | A critical ticket may be assigned low priority. | Escalation rules check urgent, fraud, breach, and security terms. |
| Privacy exposure | Ticket text may contain emails, phone numbers, or other identifiers. | Text cleaning and summaries mask obvious emails and phone numbers. |
| Historical bias | Historical routing labels may reflect inconsistent human decisions. | Macro F1, per-class, and per-language analysis are used for reflection. |
| Over-reliance | Agents may treat AI output as authoritative. | Explanation terms and confidence guidance encourage human review. |
| Data leakage | Post-response information could inflate performance. | The post-response `answer` field is excluded from prediction features. |

## Confidence-aware human review

Confidence guidance was added so uncertain predictions can be flagged for human
review. The system reports queue confidence, priority confidence, confidence
levels, and a review recommendation. This improves responsible deployment by
making uncertainty visible rather than hiding it behind a single predicted
label.

## Human-in-the-loop design

The app is designed to support agents rather than replace them. A support agent
should use the model output as an initial triage suggestion and then confirm the
queue, priority, and customer response. Escalated cases should receive faster
human review.

## Scalability considerations

The current prototype runs locally with saved model files. Production deployment
would require authentication, role-based access, audit logging, monitoring,
drift detection, periodic retraining, and stronger privacy redaction. Confidence
statistics should also be monitored over time to detect shifts in ticket
language or customer behaviour.

## Recommended future controls

- Add calibrated confidence evaluation on larger validation sets.
- Add active learning so agents can correct uncertain predictions.
- Add Unicode-aware and multilingual privacy redaction.
- Add SLA-aware escalation rules.
- Add organisation-specific testing before production use.
- Add monitoring for language-specific and class-specific performance.

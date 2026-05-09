# Presentation Demo Examples

<<<<<<< HEAD
Use these examples in the Streamlit demo. They are written to show different parts of the system: queue prediction, priority prediction, routing, summarisation, explanation terms, escalation rules, and batch upload.

Start in the `Overview` tab to show the dataset profile, workflow, model comparison chart, per-language/per-class evidence, and optional transformer benchmark results. Then switch to `Try Solution` for single-ticket or batch prediction.
=======
Use these examples in the Streamlit demo. They are written to show different parts of the system: queue prediction, priority prediction, routing, summarisation, explanation terms, and escalation rules.
>>>>>>> 1fd79b5 (Add MLP classifier, RL routing agents)

## Example 1: Security Incident

Subject: Possible unauthorised access to admin portal

Body: Several staff members received unexpected password reset emails this morning. One admin account shows login attempts from an unknown location. We need urgent help checking whether customer records or billing details were accessed.

Expected discussion points:

- likely high priority
- escalation should be triggered
- routing should go to technical or security-related support
- useful for ethics discussion because it may involve sensitive customer data

## Example 2: Billing Problem

Subject: Duplicate invoice charged to account

Body: Our company subscription was charged twice for the same billing period. The invoice numbers are different but both payments were taken from the same card. Please review the account and reverse the duplicate payment.

Expected discussion points:

- billing/payment queue
- medium or high priority depending on model output
- clear summary and routing value

## Example 3: Product Support Request

Subject: Compatibility question before upgrade

Body: We are planning to upgrade our analytics dashboard and need to confirm whether the new version supports Salesforce integration, single sign-on, and scheduled PDF exports for management reports.

Expected discussion points:

- product support or sales/pre-sales
- lower urgency than incident examples
- useful example for showing explainability terms

## Example 4: Service Outage

Subject: Dashboard unavailable for multiple users

Body: The customer reporting dashboard has been unavailable for our support team for the last hour. Clearing the browser cache and restarting devices did not fix the issue. Multiple users are blocked from viewing live case data.

Expected discussion points:

- service outage or technical support
- escalation may trigger because the issue blocks multiple users
- good example for limitations if queue prediction is uncertain

## Example 5: Returns and Exchange

Subject: Replacement request for damaged hardware

Body: We received the ordered scanner yesterday, but the screen is cracked and the device will not power on. Please arrange a replacement or return process as soon as possible.

Expected discussion points:

- returns and exchanges
- practical routing example
- simple user-facing summary
<<<<<<< HEAD

## Batch Upload Demo

For the batch workflow, create a small CSV or Excel file with a column named `ticket_text`. The app also recognises columns such as `body`, `description`, `message`, `text`, or `subject`.

Example CSV content:

```csv
ticket_text
"Subject: Possible unauthorised access to admin portal. Body: Several staff received unexpected password reset emails and one admin account shows login attempts from an unknown location."
"Subject: Duplicate invoice charged to account. Body: Our company subscription was charged twice for the same billing period and we need the duplicate payment reversed."
"Subject: Dashboard unavailable for multiple users. Body: The customer reporting dashboard has been unavailable for an hour and multiple users cannot view live case data."
```

Expected discussion points:

- shows the app can analyse multiple tickets without manually pasting each one
- output includes predicted queue, predicted priority, recommended team, escalation flag, and summary
- predictions can be downloaded as `ticket_predictions.csv`
- reinforces that the app is a decision-support tool and humans still review the final action
=======
>>>>>>> 1fd79b5 (Add MLP classifier, RL routing agents)

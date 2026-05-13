"""Repository health checks for reproducibility and submission readiness."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, RESULTS_DIR


REQUIRED_RESULT_FILES = [
    "dataset_profile.json",
    "metrics_summary.csv",
    "classification_reports.json",
    "confusion_matrix_category.png",
    "confusion_matrix_priority.png",
]


def has_conflict_markers(path: Path) -> bool:
    """Return True when a text file still contains Git conflict markers."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return any(marker in text for marker in ("<<<<<<<", "=======", ">>>>>>>"))


def validate_metrics_file(path: Path = RESULTS_DIR / "metrics_summary.csv") -> list[str]:
    """Validate the main metrics CSV used by the report and app."""
    issues: list[str] = []
    if not path.exists():
        return [f"Missing metrics file: {path}"]
    if has_conflict_markers(path):
        issues.append(f"Conflict markers remain in {path}")

    metrics = pd.read_csv(path)
    required_columns = {
        "task",
        "model",
        "split",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    }
    missing = required_columns.difference(metrics.columns)
    if missing:
        issues.append(f"Missing metrics columns: {', '.join(sorted(missing))}")
    if metrics.empty:
        issues.append("Metrics file is empty.")
    return issues


def run_health_check(root: Path = PROJECT_ROOT) -> list[str]:
    """Run lightweight repository checks that support reproducibility."""
    issues: list[str] = []
    for relative_path in [
        "README.md",
        "docs/run_checklist.md",
        "app/streamlit_app.py",
        "src/customer_support_ai/predict.py",
    ]:
        path = root / relative_path
        if not path.exists():
            issues.append(f"Missing required file: {relative_path}")
        elif has_conflict_markers(path):
            issues.append(f"Conflict markers remain in {relative_path}")

    for filename in REQUIRED_RESULT_FILES:
        if not (RESULTS_DIR / filename).exists():
            issues.append(f"Missing result artifact: results/{filename}")

    issues.extend(validate_metrics_file())
    return issues


def main() -> None:
    """CLI entry point for repository quality checks."""
    parser = argparse.ArgumentParser(description="Check repository submission readiness.")
    parser.parse_args()
    issues = run_health_check()
    if issues:
        print("Repository health check failed:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("Repository health check passed.")


if __name__ == "__main__":
    main()

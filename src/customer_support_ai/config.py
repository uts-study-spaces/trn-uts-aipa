"""Shared project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_DATASET_FILENAME = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
DEFAULT_DATASET_PATH = DATA_RAW_DIR / DEFAULT_DATASET_FILENAME
COMPATIBLE_DATASET_FILENAMES = [
    "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
    "dataset-tickets-multi-lang-4-20k.csv",
    "dataset-tickets-multi-lang3-4k.csv",
]
COMPATIBLE_DATASET_PATHS = [DATA_RAW_DIR / filename for filename in COMPATIBLE_DATASET_FILENAMES]
AUDIT_ONLY_DATASET_FILENAMES = [
    "dataset-tickets-german_normalized.csv",
    "dataset-tickets-german_normalized_50_5_2.csv",
]
KAGGLE_DATASET_HANDLE = "tobiasbueck/multilingual-customer-support-tickets"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

TEXT_COLUMN_CANDIDATES = [
    "body",
    "issue_description",
    "description",
    "ticket_description",
    "customer_issue",
    "text",
]

SUBJECT_COLUMN_CANDIDATES = [
    "subject",
    "title",
]

CATEGORY_COLUMN_CANDIDATES = [
    "queue",
    "ticket_category",
    "category",
    "issue_category",
    "ticket type",
]

PRIORITY_COLUMN_CANDIDATES = [
    "priority",
    "ticket_priority",
    "severity",
]

SAFE_METADATA_CANDIDATES = [
    "support_channel",
    "channel",
    "type",
    "language",
    "product",
    "product_service",
    "product_service_type",
    "subscription_type",
    "customer_type",
    "region",
    "customer_gender",
    "customer_segment",
    "operating_system",
    "browser",
    "payment_method",
    "preferred_contact_time",
    "tag_1",
    "tag_2",
    "tag_3",
    "tag_4",
    "tag_5",
    "tag_6",
    "tag_7",
    "tag_8",
    "tag_9",
]

LEAKAGE_COLUMN_CANDIDATES = [
    "answer",
    "resolution_notes",
    "resolution",
    "resolved_at",
    "resolution_time",
    "status",
    "agent_notes",
    "customer_satisfaction_rating",
]

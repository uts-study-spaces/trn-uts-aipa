"""Download the Kaggle customer support dataset reproducibly."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .config import COMPATIBLE_DATASET_FILENAMES, DATA_RAW_DIR, DEFAULT_DATASET_PATH, KAGGLE_DATASET_HANDLE


def find_csv_file(download_dir: Path) -> Path:
    """Select the likely dataset CSV from a downloaded Kaggle folder."""
    csv_files = sorted(download_dir.rglob("*.csv"), key=lambda path: path.stat().st_size, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in Kaggle download folder: {download_dir}")
    return csv_files[0]


def download_kaggle_dataset(
    output_path: Path = DEFAULT_DATASET_PATH,
    handle: str = KAGGLE_DATASET_HANDLE,
    force: bool = False,
) -> Path:
    """Download the dataset and copy the largest CSV to data/raw."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path

    import kagglehub

    downloaded_path = Path(kagglehub.dataset_download(handle, force_download=force))
    csv_path = find_csv_file(downloaded_path)
    shutil.copy2(csv_path, output_path)
    return output_path


def download_project_datasets(force: bool = False) -> list[Path]:
    """Download Kaggle and copy the compatible CSV files used by the project."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = [DATA_RAW_DIR / filename for filename in COMPATIBLE_DATASET_FILENAMES]
    if all(path.exists() for path in output_paths) and not force:
        return output_paths

    import kagglehub

    downloaded_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_HANDLE, force_download=force))
    copied_paths = []
    for filename in COMPATIBLE_DATASET_FILENAMES:
        matches = list(downloaded_path.rglob(filename))
        if not matches:
            raise FileNotFoundError(f"Could not find {filename} in Kaggle download folder.")
        output_path = DATA_RAW_DIR / filename
        shutil.copy2(matches[0], output_path)
        copied_paths.append(output_path)
    return copied_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle customer support dataset.")
    parser.add_argument("--force", action="store_true", help="Download again even if the CSV already exists.")
    parser.add_argument("--output", default=str(DEFAULT_DATASET_PATH), help="Where to save the CSV.")
    parser.add_argument("--single-file", action="store_true", help="Only download/copy the largest CSV.")
    args = parser.parse_args()

    if args.single_file:
        output_path = download_kaggle_dataset(Path(args.output), force=args.force)
        print(f"Dataset ready: {output_path}")
    else:
        output_paths = download_project_datasets(force=args.force)
        print("Datasets ready:")
        for output_path in output_paths:
            print(f"- {output_path}")


if __name__ == "__main__":
    main()

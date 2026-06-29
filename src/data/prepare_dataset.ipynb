"""
FAKTA - Dataset Preparation

Author  : PPTI 25
Project : FAKTA (Fact Checking AI)

This module prepares the raw dataset before any NLP preprocessing
or machine learning training.

Pipeline:
Raw Dataset
    ↓
Validate Dataset
    ↓
Remove Missing Values
    ↓
Normalize Labels
    ↓
Remove Duplicates
    ↓
Print Statistics
    ↓
Save Clean Dataset
"""

from pathlib import Path
import pandas as pd


class DatasetPreparer:
    """
    Prepare raw dataset for the next preprocessing stage.
    """

    REQUIRED_COLUMNS = ["text", "label"]

    def __init__(
        self,
        input_path="data/raw/raw_dataset.csv",
        output_path="data/processed/clean_dataset.csv",
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.df = None

    # ==========================================================
    # STEP 1
    # ==========================================================

    def load_dataset(self):
        """Load raw dataset."""

        print("=" * 60)
        print("Loading dataset...")
        print("=" * 60)

        self.df = pd.read_csv(self.input_path)

        print(f"Dataset loaded successfully.")
        print(f"Rows    : {len(self.df)}")
        print(f"Columns : {list(self.df.columns)}")

    # ==========================================================
    # STEP 2
    # ==========================================================

    def validate_columns(self):
        """Validate required columns."""

        print("\nChecking required columns...")

        missing = [
            col for col in self.REQUIRED_COLUMNS
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        print("Column validation passed.")

    # ==========================================================
    # STEP 3
    # ==========================================================

    def clean_dataset(self):
        """Basic cleaning."""

        print("\nCleaning dataset...")

        before = len(self.df)

        # remove missing values
        self.df = self.df.dropna(subset=["text", "label"])

        # remove empty text
        self.df["text"] = self.df["text"].astype(str).str.strip()
        self.df["label"] = self.df["label"].astype(str).str.strip()

        self.df = self.df[
            self.df["text"].str.len() > 0
        ]

        # normalize labels
        self.df["label"] = self.df["label"].str.lower()

        allowed = {
            "hoax": "hoax",
            "valid": "valid",
            "true": "valid",
            "real": "valid",
            "false": "hoax",
            "fake": "hoax",
            "1": "hoax",
            "0": "valid",
        }

        self.df["label"] = (
            self.df["label"]
            .map(allowed)
        )

        self.df = self.df.dropna(subset=["label"])

        after = len(self.df)

        print(f"Removed {before-after} invalid rows.")

    # ==========================================================
    # STEP 4
    # ==========================================================

    def remove_duplicates(self):
        """Remove duplicate articles."""

        print("\nRemoving duplicates...")

        before = len(self.df)

        self.df = self.df.drop_duplicates(
            subset=["text"]
        )

        after = len(self.df)

        print(f"Removed {before-after} duplicate rows.")

    # ==========================================================
    # STEP 5
    # ==========================================================

    def print_statistics(self):
        """Print dataset statistics."""

        print("\nDataset Statistics")
        print("-" * 40)

        print(f"Total samples : {len(self.df)}")

        print("\nLabel Distribution")

        print(self.df["label"].value_counts())

    # ==========================================================
    # STEP 6
    # ==========================================================

    def save_dataset(self):
        """Save cleaned dataset."""

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.df.to_csv(
            self.output_path,
            index=False,
            encoding="utf-8"
        )

        print("\nDataset saved.")

        print(self.output_path)

    # ==========================================================
    # RUN
    # ==========================================================

    def run(self):

        self.load_dataset()

        self.validate_columns()

        self.clean_dataset()

        self.remove_duplicates()

        self.print_statistics()

        self.save_dataset()


if __name__ == "__main__":

    preparer = DatasetPreparer()

    preparer.run()


"""
tokenizer.py

Responsibilities:
1. Load preprocessed train/test/validation datasets.
2. Tokenize combined_text.
3. Remove empty tokens.
4. Save tokenized datasets.

This module does NOT:
- Apply TF-IDF.
- Train models.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import re


class NLPTokenizer:

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()

        self.input_dir = (
            self.project_root / "artifacts" / "preprocessed"
        )

        self.output_dir = (
            self.project_root / "artifacts" / "tokenized"
        )

    def _load_dataset(self, filename: str) -> pd.DataFrame:
        path = self.input_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessed dataset not found: {path}"
            )

        data = pd.read_csv(path)

        if "combined_text" not in data.columns:
            raise ValueError(
                f"'combined_text' column missing in {filename}"
            )

        return data

    @staticmethod
    def _tokenize_text(text: str) -> str:
        if pd.isna(text):
            return ""

        text = str(text).lower()

        # Keep alphabetic words and remove punctuation/numbers.
        tokens = re.findall(r"\b[a-zA-Z]+\b", text)

        return " ".join(tokens)

    def _tokenize_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        tokenized = data.copy()

        tokenized["combined_text"] = (
            tokenized["combined_text"]
            .fillna("")
            .apply(self._tokenize_text)
        )

        return tokenized

    def _save_dataset(
        self,
        data: pd.DataFrame,
        filename: str,
    ) -> Path:

        self.output_dir.mkdir(parents=True, exist_ok=True)

        path = self.output_dir / filename
        data.to_csv(path, index=False)

        return path

    def initiate_tokenization(self):

        print("\nNLP TOKENIZATION")

        train_data = self._load_dataset("train.csv")
        test_data = self._load_dataset("test.csv")

        validation_path = self.input_dir / "validation.csv"
        validation_data: Optional[pd.DataFrame] = None

        if validation_path.exists():
            validation_data = self._load_dataset("validation.csv")

        train_data = self._tokenize_dataset(train_data)
        test_data = self._tokenize_dataset(test_data)

        if validation_data is not None:
            validation_data = self._tokenize_dataset(validation_data)

        train_path = self._save_dataset(train_data, "train.csv")
        test_path = self._save_dataset(test_data, "test.csv")

        if validation_data is not None:
            self._save_dataset(validation_data, "validation.csv")

        print("Tokenization completed.")
        print(f"Train: {train_path}")
        print(f"Test : {test_path}")

        return train_data, validation_data, test_data

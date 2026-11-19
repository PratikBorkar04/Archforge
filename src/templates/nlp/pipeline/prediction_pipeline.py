
"""
prediction_pipeline.py

Responsibilities:
1. Load the packaged NLP model.
2. Load the fitted TF-IDF vectorizer.
3. Load project metadata.
4. Preprocess and tokenize new text.
5. Generate predictions.
"""

from pathlib import Path
import json
import re

import joblib


class NLPPredictionPipeline:

    def __init__(self, project_root: Path):

        self.project_root = Path(project_root).resolve()

        self.model_dir = (
            self.project_root
            / "artifacts"
            / "final_model"
        )

        self.model_path = self.model_dir / "model.pkl"

        self.vectorizer_path = (
            self.model_dir / "tfidf_vectorizer.pkl"
        )

        self.metadata_path = (
            self.model_dir / "metadata.json"
        )

        self.model = None
        self.vectorizer = None
        self.metadata = None

        self._load_model_package()

    def _load_model_package(self):

        print("\nLoading NLP model package...")

        required_files = [
            self.model_path,
            self.vectorizer_path,
            self.metadata_path,
        ]

        for file_path in required_files:

            if not file_path.exists():

                raise FileNotFoundError(
                    f"Required model file not found: {file_path}"
                )

        self.model = joblib.load(self.model_path)

        self.vectorizer = joblib.load(
            self.vectorizer_path
        )

        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            self.metadata = json.load(file)

        print("Model loaded successfully.")
        print("Vectorizer loaded successfully.")

    @staticmethod
    def _preprocess_text(text: str) -> str:

        if not isinstance(text, str):

            raise TypeError(
                "Input text must be a string."
            )

        # Match training preprocessing.
        text = text.lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        # Match training tokenizer.
        tokens = re.findall(
            r"\b[a-zA-Z]+\b",
            text,
        )

        return " ".join(tokens)

    def predict(self, text: str):

        if not text or not text.strip():

            raise ValueError(
                "Input text cannot be empty."
            )

        processed_text = self._preprocess_text(text)

        if not processed_text:

            raise ValueError(
                "No valid English alphabetic tokens found."
            )

        # Transform using the existing fitted vectorizer.
        text_vector = self.vectorizer.transform(
            [processed_text]
        )

        # Generate prediction.
        prediction = self.model.predict(
            text_vector
        )[0]

        result = {
            "prediction": str(prediction),
            "target_column": self.metadata.get(
                "target_column"
            ),
            "input_text": text,
        }

        # Include probability when supported by the model.
        if hasattr(self.model, "predict_proba"):

            probabilities = self.model.predict_proba(
                text_vector
            )[0]

            classes = self.model.classes_

            result["confidence"] = float(
                max(probabilities)
            )

            result["class_probabilities"] = {
                str(label): float(probability)
                for label, probability in zip(
                    classes,
                    probabilities,
                )
            }

        return result

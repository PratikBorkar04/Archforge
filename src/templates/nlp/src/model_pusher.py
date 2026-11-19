
"""
model_pusher.py

Responsibilities:
1. Load the trained model.
2. Load the fitted TF-IDF vectorizer.
3. Read the training report.
4. Package the model and vectorizer.
5. Save dynamic metadata for the prediction pipeline and UI.
"""

from pathlib import Path
import json
import shutil


class NLPModelPusher:

    def __init__(
        self,
        project_root: Path,
        target_column: str,
        text_columns: list,
    ) -> None:

        self.project_root = Path(project_root).resolve()

        self.model_path = (
            self.project_root
            / "artifacts"
            / "models"
            / "best_model.pkl"
        )

        self.vectorizer_path = (
            self.project_root
            / "artifacts"
            / "transformed"
            / "tfidf_vectorizer.pkl"
        )

        self.training_report_path = (
            self.project_root
            / "artifacts"
            / "models"
            / "training_report.json"
        )

        self.output_dir = (
            self.project_root
            / "artifacts"
            / "final_model"
        )

        self.target_column = target_column
        self.text_columns = text_columns

    def _load_training_report(self):

        if not self.training_report_path.exists():
            raise FileNotFoundError(
                f"Training report not found: "
                f"{self.training_report_path}"
            )

        with self.training_report_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            report = json.load(file)

        best_model_name = report.get("best_model")
        model_results = report.get("models", {})

        if not best_model_name:
            raise ValueError(
                "Best model name missing from training report."
            )

        if best_model_name not in model_results:
            raise ValueError(
                "Best model metrics missing from training report."
            )

        best_model_metrics = model_results[best_model_name]

        if "accuracy" not in best_model_metrics:
            raise ValueError(
                "Best model validation accuracy is missing."
            )

        if "f1_score" not in best_model_metrics:
            raise ValueError(
                "Best model weighted F1-score is missing."
            )

        return report, best_model_name, best_model_metrics

    def initiate_model_pusher(self):

        print("\nNLP MODEL PUSHER")

        # Check trained model.
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}"
            )

        # Check fitted TF-IDF vectorizer.
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found: "
                f"{self.vectorizer_path}"
            )

        # Load training report and selected model metrics.
        (
            training_report,
            best_model_name,
            best_model_metrics,
        ) = self._load_training_report()

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Copy trained model.
        final_model_path = (
            self.output_dir / "model.pkl"
        )

        shutil.copy2(
            self.model_path,
            final_model_path,
        )

        # Copy fitted TF-IDF vectorizer.
        final_vectorizer_path = (
            self.output_dir / "tfidf_vectorizer.pkl"
        )

        shutil.copy2(
            self.vectorizer_path,
            final_vectorizer_path,
        )

        # Save dynamic metadata.
        metadata = {
            "target_column": self.target_column,
            "text_columns": self.text_columns,

            "model_file": "model.pkl",
            "vectorizer_file": "tfidf_vectorizer.pkl",

            "best_model": best_model_name,

            "validation_accuracy": best_model_metrics["accuracy"],

            "validation_f1_score": best_model_metrics["f1_score"],

            "selection_metric": training_report.get(
                "selection_metric"
            ),

            "validation_method": training_report.get(
                "validation_method"
            ),

            "model_results": training_report.get(
                "models", {}
            ),
        }

        metadata_path = (
            self.output_dir / "metadata.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4,
            )

        print("\nModel packaging completed.")

        print(f"Best model          : {best_model_name}")
        print(
            f"Validation accuracy : "
            f"{best_model_metrics['accuracy']:.4f}"
        )
        print(
            f"Validation F1-score : "
            f"{best_model_metrics['f1_score']:.4f}"
        )

        print(f"\nModel      : {final_model_path}")
        print(f"Vectorizer : {final_vectorizer_path}")
        print(f"Metadata   : {metadata_path}")

        print("\nFinal model directory:")
        print(self.output_dir)

        return (
            final_model_path,
            final_vectorizer_path,
            metadata_path,
        )

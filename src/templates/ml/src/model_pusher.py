"""
model_pusher.py

Model saving component for ArchForge.

Responsibilities
----------------
1. Receive the best trained model from ModelEvaluation.
2. Save the model as a serialized artifact.
3. Save model metadata.
4. Save feature and target information.
5. Save information about the evaluation used to select the model.
6. Return paths to the saved artifacts.

This module does NOT:
- Train models.
- Evaluate models.
- Select models.

Those responsibilities belong to:
- model_trainer.py
- model_evaluation.py
"""

from pathlib import Path
from typing import Dict, Any

import json
import pickle

import numpy as np
import pandas as pd


class ModelPusher:
    """
    Save the best-performing trained model and its metadata.
    """

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(
        self,
        project_root: Path,
    ) -> None:

        self.project_root = Path(
            project_root
        ).resolve()

        self.model_dir = (
            self.project_root
            / "artifacts"
            / "models"
        )

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ============================================================
    # Main Model Pushing Method
    # ============================================================

    def initiate_model_pushing(
        self,
        best_model: Any,
        best_model_name: str,
        best_model_score: float,
        primary_metric: str,
        problem_type: str,
        evaluation_source: str,
        evaluation_samples: int,
        evaluation_report: Dict[str, Any] = None,
        target_column: str = None,
        feature_columns: list = None,
    ) -> Dict[str, Any]:
        """
        Save the best model and its metadata.

        Parameters
        ----------
        best_model:
            Trained model selected by ModelEvaluation.

        best_model_name:
            Name of the selected model.

        best_model_score:
            Best model's primary evaluation score.

        primary_metric:
            Metric used for model selection.

        problem_type:
            REGRESSION or CLASSIFICATION.

        evaluation_source:
            validation or test.

        evaluation_samples:
            Number of samples used during evaluation.

        evaluation_report:
            Complete evaluation report, when available.

        target_column:
            Name of the target column.

        feature_columns:
            Original feature column names used by the model.

        Returns
        -------
        Dict[str, Any]
            Information about saved model artifacts.
        """

        # --------------------------------------------------------
        # Validate inputs
        # --------------------------------------------------------

        if best_model is None:

            raise ValueError(
                "Best model cannot be None."
            )

        if not best_model_name:

            raise ValueError(
                "Best model name is required."
            )

        if not primary_metric:

            raise ValueError(
                "Primary metric is required."
            )

        if not problem_type:

            raise ValueError(
                "Problem type is required."
            )

        if not target_column:

            raise ValueError(
                "Target column is required."
            )

        if not feature_columns:

            raise ValueError(
                "Feature columns are required."
            )

        # --------------------------------------------------------
        # Normalize values
        # --------------------------------------------------------

        problem_type = (
            str(problem_type)
            .strip()
            .upper()
        )

        evaluation_source = (
            str(evaluation_source)
            .strip()
            .lower()
        )

        target_column = (
            str(target_column)
            .strip()
        )

        feature_columns = [
            str(column).strip()
            for column in feature_columns
        ]

        # --------------------------------------------------------
        # File paths
        # --------------------------------------------------------

        model_path = (
            self.model_dir
            / "best_model.pkl"
        )

        metadata_path = (
            self.model_dir
            / "model_metadata.json"
        )

        # --------------------------------------------------------
        # Save model
        # --------------------------------------------------------

        print(
            "\n" + "=" * 60
        )

        print(
            "MODEL SAVING"
        )

        print(
            "=" * 60
        )

        print(
            f"Model: {best_model_name}"
        )

        print(
            f"Primary metric: {primary_metric}"
        )

        print(
            f"Score: "
            f"{self._format_score(best_model_score)}"
        )

        print(
            f"Evaluation source: "
            f"{evaluation_source}"
        )

        print(
            f"Target column: "
            f"{target_column}"
        )

        print(
            f"Input features: "
            f"{len(feature_columns)}"
        )

        print(
            "\nSaving trained model..."
        )

        with open(
            model_path,
            "wb",
        ) as file:

            pickle.dump(
                best_model,
                file,
            )

        print(
            "✓ Model saved:"
        )

        print(
            f"  {model_path}"
        )

        # --------------------------------------------------------
        # Build metadata
        # --------------------------------------------------------

        metadata = {

            "model_name":
                best_model_name,

            "problem_type":
                problem_type,

            "target_column":
                target_column,

            "features":
                feature_columns,

            "feature_count":
                len(feature_columns),

            "primary_metric":
                primary_metric,

            "best_model_score":
                self._safe_float(
                    best_model_score
                ),

            "evaluation_source":
                evaluation_source,

            "evaluation_samples":
                int(
                    evaluation_samples
                ),

            "model_class":
                best_model.__class__.__name__,

            "model_module":
                best_model.__class__.__module__,

            "model_path":
                str(model_path),

        }

        # --------------------------------------------------------
        # Add evaluation summary
        # --------------------------------------------------------

        if evaluation_report:

            metadata[
                "evaluation_summary"
            ] = {

                "primary_metric":
                    evaluation_report.get(
                        "primary_metric"
                    ),

                "best_model":
                    evaluation_report.get(
                        "best_model"
                    ),

                "best_model_score":
                    self._safe_float(
                        evaluation_report.get(
                            "best_model_score"
                        )
                    ),

            }

        # --------------------------------------------------------
        # Save metadata
        # --------------------------------------------------------

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
                default=self._json_converter,
            )

        print(
            "\n✓ Model metadata saved:"
        )

        print(
            f"  {metadata_path}"
        )

        # --------------------------------------------------------
        # Summary
        # --------------------------------------------------------

        print(
            "\n" + "=" * 60
        )

        print(
            "MODEL SAVING SUMMARY"
        )

        print(
            "=" * 60
        )

        print(
            f"Model name       : "
            f"{best_model_name}"
        )

        print(
            f"Model type       : "
            f"{best_model.__class__.__name__}"
        )

        print(
            f"Problem type     : "
            f"{problem_type}"
        )

        print(
            f"Target column    : "
            f"{target_column}"
        )

        print(
            f"Input features   : "
            f"{len(feature_columns)}"
        )

        for column in feature_columns:

            print(
                f"  • {column}"
            )

        print(
            f"Primary metric   : "
            f"{primary_metric}"
        )

        print(
            f"Score            : "
            f"{self._format_score(best_model_score)}"
        )

        print(
            f"Evaluation source: "
            f"{evaluation_source}"
        )

        print(
            f"Model artifact   : "
            f"{model_path}"
        )

        print(
            f"Metadata artifact: "
            f"{metadata_path}"
        )

        print(
            "\nModel saving completed successfully."
        )

        # --------------------------------------------------------
        # Return results
        # --------------------------------------------------------

        return {

            "status":
                "SUCCESS",

            "best_model":
                best_model,

            "best_model_name":
                best_model_name,

            "model_path":
                model_path,

            "metadata_path":
                metadata_path,

            "metadata":
                metadata,
        }

    # ============================================================
    # Score Formatting
    # ============================================================

    @staticmethod
    def _format_score(
        score: Any,
    ) -> str:
        """
        Safely format model score for terminal output.
        """

        try:

            if score is None:

                return "N/A"

            if isinstance(
                score,
                float,
            ) and np.isnan(score):

                return "NaN"

            return f"{float(score):.4f}"

        except (
            TypeError,
            ValueError,
        ):

            return str(score)

    # ============================================================
    # Safe Float Conversion
    # ============================================================

    @staticmethod
    def _safe_float(
        value: Any,
    ):
        """
        Convert numeric values to JSON-safe Python floats.
        """

        if value is None:

            return None

        try:

            numeric_value = float(
                value
            )

            if np.isnan(
                numeric_value
            ):

                return None

            if np.isinf(
                numeric_value
            ):

                return None

            return numeric_value

        except (
            TypeError,
            ValueError,
        ):

            return value

    # ============================================================
    # JSON Converter
    # ============================================================

    @staticmethod
    def _json_converter(
        value: Any,
    ):
        """
        Convert NumPy and Pandas values
        into JSON-compatible values.
        """

        if isinstance(
            value,
            (
                np.integer,
                np.floating,
            ),
        ):

            return value.item()

        if isinstance(
            value,
            np.ndarray,
        ):

            return value.tolist()

        if isinstance(
            value,
            (
                pd.Series,
                pd.Index,
            ),
        ):

            return value.tolist()

        if isinstance(
            value,
            Path,
        ):

            return str(value)

        return value

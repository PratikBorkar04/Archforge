"""
model_pusher.py

Model saving component for ArchForge.

Responsibilities
----------------
1. Receive the best trained model from ModelEvaluation.
2. Save the trained model.
3. Save model metadata.
4. Save the target column.
5. Save the original feature columns.
6. Preserve feature ordering.
7. Save evaluation information.
8. Provide prediction-time schema information.

Important
---------
The target column is NEVER assumed to be the last column.

The exact target_column detected/selected by the training pipeline
is used to determine the feature columns.

This allows ArchForge to work with datasets such as:

    ID | Selling_Price | Year | Present_Price | ...

or:

    Year | Area | Bedrooms | Price | Location

or:

    Price | Feature_A | Feature_B | Feature_C

The target may appear anywhere in the dataframe.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

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

        self.project_root = (
            Path(project_root).resolve()
        )

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
        evaluation_report: Optional[
            Dict[str, Any]
        ] = None,
        target_column: Optional[str] = None,
        feature_columns: Optional[
            List[str]
        ] = None,
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
            Best model evaluation score.

        primary_metric:
            Metric used for model selection.

        problem_type:
            REGRESSION or CLASSIFICATION.

        evaluation_source:
            validation or test.

        evaluation_samples:
            Number of samples used for evaluation.

        evaluation_report:
            Complete evaluation report.

        target_column:
            Exact target column detected/selected during training.

        feature_columns:
            Original feature columns used to train the model.

            IMPORTANT:
            The target column can appear anywhere in the dataset.
            feature_columns must already exclude target_column.

        Returns
        -------
        Dict[str, Any]
            Information about saved model artifacts.
        """

        # ========================================================
        # Validate inputs
        # ========================================================

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

        # ========================================================
        # Normalize values
        # ========================================================

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

        target_column = str(
            target_column
        ).strip()

        feature_columns = [
            str(column).strip()
            for column in feature_columns
        ]

        # ========================================================
        # Validate feature/target relationship
        # ========================================================

        if target_column in feature_columns:

            raise ValueError(
                "Target column was included in "
                "feature_columns. This would cause "
                "target leakage."
            )

        if len(feature_columns) == 0:

            raise ValueError(
                "No feature columns remain after "
                "removing the target column."
            )

        # ========================================================
        # File paths
        # ========================================================

        model_path = (
            self.model_dir
            / "best_model.pkl"
        )

        metadata_path = (
            self.model_dir
            / "model_metadata.json"
        )

        # ========================================================
        # Console header
        # ========================================================

        print(
            "\n"
            + "=" * 60
        )

        print(
            "MODEL SAVING"
        )

        print(
            "=" * 60
        )

        print(
            f"Model              : "
            f"{best_model_name}"
        )

        print(
            f"Problem type       : "
            f"{problem_type}"
        )

        print(
            f"Target column      : "
            f"{target_column}"
        )

        print(
            f"Input feature count: "
            f"{len(feature_columns)}"
        )

        print(
            f"Primary metric     : "
            f"{primary_metric}"
        )

        print(
            f"Score              : "
            f"{self._format_score(best_model_score)}"
        )

        print(
            f"Evaluation source  : "
            f"{evaluation_source}"
        )

        # ========================================================
        # Save model
        # ========================================================

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

        # ========================================================
        # Build metadata
        # ========================================================

        metadata = {

            # ----------------------------------------------------
            # Model information
            # ----------------------------------------------------

            "model_name":
                best_model_name,

            "model_class":
                best_model.__class__.__name__,

            "model_module":
                best_model.__class__.__module__,

            # ----------------------------------------------------
            # Problem information
            # ----------------------------------------------------

            "problem_type":
                problem_type,

            # ----------------------------------------------------
            # Target information
            # ----------------------------------------------------

            "target_column":
                target_column,

            # ----------------------------------------------------
            # Feature information
            # ----------------------------------------------------

            "features":
                feature_columns,

            "feature_columns":
                feature_columns,

            "feature_count":
                len(feature_columns),

            # ----------------------------------------------------
            # Explicit prediction schema
            # ----------------------------------------------------

            "prediction_schema": {

                "target_column":
                    target_column,

                "feature_columns":
                    feature_columns,

                "feature_count":
                    len(feature_columns),

            },

            # ----------------------------------------------------
            # Evaluation information
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # Artifact information
            # ----------------------------------------------------

            "model_path":
                str(model_path),

        }

        # ========================================================
        # Add evaluation summary
        # ========================================================

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

        # ========================================================
        # Save metadata
        # ========================================================

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
                allow_nan=False,
            )

        print(
            "\n✓ Model metadata saved:"
        )

        print(
            f"  {metadata_path}"
        )

        # ========================================================
        # Summary
        # ========================================================

        print(
            "\n"
            + "=" * 60
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

        print(
            "\nFeature columns:"
        )

        for index, column in enumerate(
            feature_columns,
            start=1,
        ):

            print(
                f"  {index}. {column}"
            )

        print(
            f"\nPrimary metric   : "
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

        # ========================================================
        # Return results
        # ========================================================

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

            "target_column":
                target_column,

            "feature_columns":
                feature_columns,

            "feature_count":
                len(feature_columns),

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

            numeric_score = float(
                score
            )

            if np.isnan(
                numeric_score
            ):

                return "NaN"

            if np.isinf(
                numeric_score
            ):

                return "Inf"

            return f"{numeric_score:.4f}"

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
        Convert numeric values into JSON-safe
        Python floats.
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
        Convert NumPy and Pandas objects
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
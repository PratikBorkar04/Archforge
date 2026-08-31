"""
model_evaluation.py

Model evaluation component for ArchForge.

Responsibilities
----------------
1. Receive trained models.
2. Evaluate models on validation data when available.
3. Fall back to test data when validation data is unavailable.
4. Automatically handle regression and classification.
5. Calculate appropriate evaluation metrics.
6. Rank trained models.
7. Select the best-performing model.
8. Save evaluation results.

This module does NOT:
- Train models.
- Select candidate models before training.
- Save the final model artifact.
"""

from pathlib import Path
from typing import Dict, Any

import json
import math

import numpy as np
import pandas as pd

from sklearn.metrics import (
    # Regression
    mean_absolute_error,
    mean_squared_error,
    r2_score,

    # Classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class ModelEvaluation:
    """
    Evaluate trained models and select the best model.
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

        self.evaluation_dir = (
            self.project_root
            / "artifacts"
            / "evaluation"
        )

        self.evaluation_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ============================================================
    # Main Evaluation Method
    # ============================================================

    def initiate_model_evaluation(
        self,
        trained_models: Dict[str, Any],
        transformed_data: Any,
        target_column: str,
        problem_type: str,
    ) -> Dict[str, Any]:
        """
        Evaluate all successfully trained models.

        Validation data is preferred.
        Test data is used only when validation data
        is unavailable.

        Parameters
        ----------
        trained_models:
            Dictionary containing trained model objects.

        transformed_data:
            Output from DataTransformation.

        target_column:
            Target column name.

        problem_type:
            REGRESSION or CLASSIFICATION.

        Returns
        -------
        Dict[str, Any]
            Evaluation results and best model.
        """

        # --------------------------------------------------------
        # Input validation
        # --------------------------------------------------------

        if not trained_models:

            raise ValueError(
                "No trained models available "
                "for evaluation."
            )

        if not target_column:

            raise ValueError(
                "Target column is required."
            )

        if not problem_type:

            raise ValueError(
                "Problem type is required."
            )

        problem_type = (
            str(problem_type)
            .strip()
            .upper()
        )

        if problem_type not in {
            "REGRESSION",
            "CLASSIFICATION",
        }:

            raise ValueError(
                f"Unsupported problem type: "
                f"{problem_type}"
            )

        # --------------------------------------------------------
        # Prepare evaluation data
        # --------------------------------------------------------

        (
            X_eval,
            y_eval,
            evaluation_source,
        ) = self._prepare_evaluation_data(
            transformed_data=transformed_data,
            target_column=target_column,
        )

        if X_eval is None or y_eval is None:

            raise ValueError(
                "Unable to prepare evaluation data."
            )

        if len(X_eval) == 0:

            raise ValueError(
                "Evaluation dataset is empty."
            )

        # --------------------------------------------------------
        # Concise output
        # --------------------------------------------------------

        print(
            f"Evaluating {len(trained_models)} "
            f"model(s) on {len(X_eval)} samples "
            f"({evaluation_source})..."
        )

        # --------------------------------------------------------
        # Evaluate models
        # --------------------------------------------------------

        evaluation_results = {}

        for model_name, model in trained_models.items():

            try:

                predictions = model.predict(
                    X_eval
                )

                # ------------------------------------------------
                # Regression
                # ------------------------------------------------

                if problem_type == "REGRESSION":

                    metrics = (
                        self._evaluate_regression(
                            y_true=y_eval,
                            y_pred=predictions,
                        )
                    )

                    primary_metric = "r2_score"

                # ------------------------------------------------
                # Classification
                # ------------------------------------------------

                else:

                    metrics = (
                        self._evaluate_classification(
                            y_true=y_eval,
                            y_pred=predictions,
                        )
                    )

                    primary_metric = "f1_score"

                primary_score = metrics.get(
                    primary_metric
                )

                # ------------------------------------------------
                # Invalid score protection
                # ------------------------------------------------

                if (
                    primary_score is None
                    or not np.isfinite(
                        primary_score
                    )
                ):

                    raise ValueError(
                        f"Primary metric "
                        f"{primary_metric} "
                        f"is not a valid finite value."
                    )

                evaluation_results[
                    model_name
                ] = {

                    "status":
                        "SUCCESS",

                    "metrics":
                        metrics,

                    "primary_metric":
                        primary_metric,

                    "primary_score":
                        float(
                            primary_score
                        ),
                }

            except Exception as exc:

                evaluation_results[
                    model_name
                ] = {

                    "status":
                        "FAILED",

                    "metrics":
                        {},

                    "error":
                        str(exc),
                }

        # --------------------------------------------------------
        # Find successful evaluations
        # --------------------------------------------------------

        successful_results = {

            name: result

            for name, result
            in evaluation_results.items()

            if result[
                "status"
            ] == "SUCCESS"

        }

        failed_results = {

            name: result

            for name, result
            in evaluation_results.items()

            if result[
                "status"
            ] == "FAILED"

        }

        if not successful_results:

            raise RuntimeError(
                "All trained models failed "
                "during evaluation."
            )

        # --------------------------------------------------------
        # Rank models
        # --------------------------------------------------------

        ranked_models = sorted(
            successful_results.items(),
            key=lambda item:
                item[1][
                    "primary_score"
                ],
            reverse=True,
        )

        # --------------------------------------------------------
        # Create ranking
        # --------------------------------------------------------

        ranking = []

        for rank, (
            model_name,
            result,
        ) in enumerate(
            ranked_models,
            start=1,
        ):

            ranking.append({

                "rank":
                    rank,

                "model":
                    model_name,

                "primary_metric":
                    result[
                        "primary_metric"
                    ],

                "primary_score":
                    result[
                        "primary_score"
                    ],

                "metrics":
                    result[
                        "metrics"
                    ],
            })

        # --------------------------------------------------------
        # Best model
        # --------------------------------------------------------

        best_model_name = (
            ranked_models[0][0]
        )

        best_model = (
            trained_models[
                best_model_name
            ]
        )

        best_result = (
            evaluation_results[
                best_model_name
            ]
        )

        # --------------------------------------------------------
        # Build evaluation report
        # --------------------------------------------------------

        report = {

            "problem_type":
                problem_type,

            "evaluation_source":
                evaluation_source,

            "evaluation_samples":
                int(
                    len(X_eval)
                ),

            "evaluation_features":
                int(
                    X_eval.shape[1]
                    if hasattr(
                        X_eval,
                        "shape",
                    )
                    and len(
                        X_eval.shape
                    ) > 1
                    else 0
                ),

            "primary_metric":
                best_result[
                    "primary_metric"
                ],

            "best_model":
                best_model_name,

            "best_model_score":
                best_result[
                    "primary_score"
                ],

            "models":
                evaluation_results,

            "ranking":
                ranking,
        }

        # --------------------------------------------------------
        # Save report
        # --------------------------------------------------------

        report_path = (
            self.evaluation_dir
            / "evaluation_report.json"
        )

        self._save_report(
            report=report,
            path=report_path,
        )

        # --------------------------------------------------------
        # Concise terminal output
        # --------------------------------------------------------

        print(
            f"Evaluation completed: "
            f"{len(successful_results)} succeeded, "
            f"{len(failed_results)} failed."
        )

        print(
            f"Best model: "
            f"{best_model_name} "
            f"({best_result['primary_metric']}="
            f"{best_result['primary_score']:.4f})"
        )

        print(
            f"Evaluation report saved: "
            f"{report_path}"
        )

        # --------------------------------------------------------
        # Return results
        # --------------------------------------------------------

        return {

            "best_model":
                best_model,

            "best_model_name":
                best_model_name,

            "best_model_score":
                best_result[
                    "primary_score"
                ],

            "primary_metric":
                best_result[
                    "primary_metric"
                ],

            "evaluation_results":
                evaluation_results,

            "ranking":
                ranking,

            "evaluation_source":
                evaluation_source,

            "evaluation_samples":
                int(
                    len(X_eval)
                ),

            "evaluation_report":
                report,

            "evaluation_report_path":
                report_path,
        }

    # ============================================================
    # Prepare Evaluation Data
    # ============================================================

    @staticmethod
    def _prepare_evaluation_data(
        transformed_data: Any,
        target_column: str,
    ):
        """
        Extract validation/test data from transformed_data.

        Priority:

        1. Validation data
        2. Test data

        Supported structures include dictionaries,
        tuples/lists and nested structures.
        """

        if transformed_data is None:

            raise ValueError(
                "Transformed data cannot be None."
            )

        # --------------------------------------------------------
        # Dictionary
        # --------------------------------------------------------

        if isinstance(
            transformed_data,
            dict,
        ):

            # ----------------------------------------------------
            # Validation data
            # ----------------------------------------------------

            validation_pairs = [

                (
                    "X_validation",
                    "y_validation",
                ),

                (
                    "validation_features",
                    "validation_target",
                ),

                (
                    "X_val",
                    "y_val",
                ),

            ]

            for X_key, y_key in validation_pairs:

                if (
                    X_key in transformed_data
                    and
                    y_key in transformed_data
                ):

                    X_validation = (
                        transformed_data[
                            X_key
                        ]
                    )

                    y_validation = (
                        transformed_data[
                            y_key
                        ]
                    )

                    if (
                        X_validation is not None
                        and
                        y_validation is not None
                        and
                        len(X_validation) > 0
                    ):

                        return (
                            X_validation,
                            y_validation,
                            "validation",
                        )

            # ----------------------------------------------------
            # Nested validation data
            # ----------------------------------------------------

            if (
                "validation_data"
                in transformed_data
            ):

                result = (
                    ModelEvaluation
                    ._extract_pair(
                        transformed_data[
                            "validation_data"
                        ],
                        target_column,
                    )
                )

                if result is not None:

                    if (
                        len(result[0])
                        > 0
                    ):

                        return (
                            result[0],
                            result[1],
                            "validation",
                        )

            # ----------------------------------------------------
            # Test fallback
            # ----------------------------------------------------

            test_pairs = [

                (
                    "X_test",
                    "y_test",
                ),

                (
                    "test_features",
                    "test_target",
                ),

            ]

            for X_key, y_key in test_pairs:

                if (
                    X_key in transformed_data
                    and
                    y_key in transformed_data
                ):

                    X_test = (
                        transformed_data[
                            X_key
                        ]
                    )

                    y_test = (
                        transformed_data[
                            y_key
                        ]
                    )

                    if (
                        X_test is not None
                        and
                        y_test is not None
                        and
                        len(X_test) > 0
                    ):

                        return (
                            X_test,
                            y_test,
                            "test",
                        )

            # ----------------------------------------------------
            # Nested test data
            # ----------------------------------------------------

            if (
                "test_data"
                in transformed_data
            ):

                result = (
                    ModelEvaluation
                    ._extract_pair(
                        transformed_data[
                            "test_data"
                        ],
                        target_column,
                    )
                )

                if result is not None:

                    if (
                        len(result[0])
                        > 0
                    ):

                        return (
                            result[0],
                            result[1],
                            "test",
                        )

        # --------------------------------------------------------
        # Tuple / list
        # --------------------------------------------------------

        if isinstance(
            transformed_data,
            (tuple, list),
        ):

            if len(transformed_data) >= 3:

                validation_result = (
                    ModelEvaluation
                    ._extract_pair(
                        transformed_data[1],
                        target_column,
                    )
                )

                if validation_result is not None:

                    if (
                        len(
                            validation_result[0]
                        ) > 0
                    ):

                        return (
                            validation_result[0],
                            validation_result[1],
                            "validation",
                        )

                test_result = (
                    ModelEvaluation
                    ._extract_pair(
                        transformed_data[2],
                        target_column,
                    )
                )

                if test_result is not None:

                    if (
                        len(
                            test_result[0]
                        ) > 0
                    ):

                        return (
                            test_result[0],
                            test_result[1],
                            "test",
                        )

        raise ValueError(
            "Transformed data does not contain "
            "usable validation or test data."
        )

    # ============================================================
    # Extract Pair
    # ============================================================

    @staticmethod
    def _extract_pair(
        data: Any,
        target_column: str,
    ):
        """
        Extract X and y from a supported data structure.
        """

        if data is None:
            return None

        # --------------------------------------------------------
        # Dictionary
        # --------------------------------------------------------

        if isinstance(
            data,
            dict,
        ):

            pairs = [

                (
                    "X_validation",
                    "y_validation",
                ),

                (
                    "X_test",
                    "y_test",
                ),

                (
                    "validation_features",
                    "validation_target",
                ),

                (
                    "test_features",
                    "test_target",
                ),

                (
                    "X",
                    "y",
                ),

            ]

            for X_key, y_key in pairs:

                if (
                    X_key in data
                    and
                    y_key in data
                ):

                    return (
                        data[X_key],
                        data[y_key],
                    )

        # --------------------------------------------------------
        # Tuple / list
        # --------------------------------------------------------

        if isinstance(
            data,
            (tuple, list),
        ):

            if len(data) >= 2:

                return (
                    data[0],
                    data[1],
                )

        # --------------------------------------------------------
        # DataFrame
        # --------------------------------------------------------

        if hasattr(
            data,
            "columns",
        ):

            if (
                target_column
                in data.columns
            ):

                X = data.drop(
                    columns=[
                        target_column
                    ]
                )

                y = data[
                    target_column
                ]

                return (
                    X,
                    y,
                )

        return None

    # ============================================================
    # Regression Evaluation
    # ============================================================

    @staticmethod
    def _evaluate_regression(
        y_true,
        y_pred,
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.

        R² requires at least two evaluation samples.
        """

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        mse = mean_squared_error(
            y_true,
            y_pred,
        )

        rmse = math.sqrt(
            mse
        )

        # --------------------------------------------------------
        # R² protection
        # --------------------------------------------------------

        if len(y_true) >= 2:

            r2 = r2_score(
                y_true,
                y_pred,
            )

        else:

            r2 = float("nan")

        return {

            "mae":
                float(mae),

            "mse":
                float(mse),

            "rmse":
                float(rmse),

            "r2_score":
                float(r2),
        }

    # ============================================================
    # Classification Evaluation
    # ============================================================

    @staticmethod
    def _evaluate_classification(
        y_true,
        y_pred,
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        """

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        return {

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1_score":
                float(f1),
        }

    # ============================================================
    # Save Report
    # ============================================================

    @staticmethod
    def _save_report(
        report: Dict[str, Any],
        path: Path,
    ) -> None:
        """
        Save evaluation report as JSON.
        """

        def convert(value):

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
                Path,
            ):

                return str(value)

            return value

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
                default=convert,
                allow_nan=False,
            )

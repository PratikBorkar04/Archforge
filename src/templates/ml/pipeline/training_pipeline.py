"""
training_pipeline.py

Main orchestration pipeline for ArchForge ML projects.

Pipeline
--------
1. Data Ingestion
2. Data Validation
3. Dataset Profiling
4. Data Cleaning
5. Data Transformation
6. Model Selection
7. Model Training
8. Model Evaluation
9. Model Pushing & Saving
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation
from src.dataset_profiler import DatasetProfiler
from src.data_cleaning import DataCleaning
from src.data_transformation import DataTransformation
from src.model_selector import ModelSelector
from src.model_trainer import ModelTrainer
from src.model_evaluation import ModelEvaluation
from src.model_pusher import ModelPusher


class TrainingPipeline:
    """
    Orchestrates the complete ML training workflow.
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

    # ============================================================
    # Run Pipeline
    # ============================================================

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete ML pipeline.
        """

        print("\n" + "=" * 60)
        print("Starting Machine Learning Pipeline")
        print("=" * 60)

        # ========================================================
        # 1. DATA INGESTION
        # ========================================================

        print("\n[1/9] Data Ingestion")

        data_ingestion = DataIngestion(
            project_root=self.project_root
        )

        (
            train_data,
            validation_data,
            test_data,
        ) = data_ingestion.initiate_data_ingestion()

        # ========================================================
        # 2. DATA VALIDATION
        # ========================================================

        print("\n[2/9] Data Validation")

        data_validation = DataValidation(
            project_root=self.project_root
        )

        validation_report = (
            data_validation.initiate_data_validation(
                train_data=train_data,
                validation_data=validation_data,
                test_data=test_data,
            )
        )

        # ========================================================
        # 3. DATASET PROFILING
        # ========================================================

        print("\n[3/9] Dataset Profiling")

        target_column = (
            self._detect_target_column(
                train_data
            )
        )

        print(
            f"Target column: {target_column}"
        )

        profiler = DatasetProfiler(
            project_root=self.project_root,
            target_column=target_column,
        )

        dataset_profile = profiler.profile(
            train_data
        )

        # ========================================================
        # 4. DATA CLEANING
        # ========================================================

        print("\n[4/9] Data Cleaning")

        data_cleaning = DataCleaning(
            project_root=self.project_root,
            target_column=target_column,
        )

        (
            cleaned_train_data,
            cleaned_validation_data,
            cleaned_test_data,
            cleaning_report,
        ) = data_cleaning.initiate_data_cleaning(
            train_data=train_data,
            validation_data=validation_data,
            test_data=test_data,
        )

        # ========================================================
        # 5. DATA TRANSFORMATION
        # ========================================================

        print("\n[5/9] Data Transformation")

        data_transformation = DataTransformation(
            project_root=self.project_root
        )

        transformed_data = (
            data_transformation
            .initiate_data_transformation(
                train_data=cleaned_train_data,
                validation_data=cleaned_validation_data,
                test_data=cleaned_test_data,
            )
        )

        # ========================================================
        # 6. MODEL SELECTION
        # ========================================================

        print("\n[6/9] Model Selection")

        model_selector = ModelSelector(
            project_root=self.project_root
        )

        model_selection = (
            model_selector.select_models(
                dataset_profile=dataset_profile
            )
        )

        # ========================================================
        # 7. MODEL TRAINING
        # ========================================================

        print("\n[7/9] Model Training")

        model_trainer = ModelTrainer(
            project_root=self.project_root
        )

        model_training = (
            model_trainer.initiate_model_training(
                train_data=transformed_data,
                selected_models=model_selection[
                    "selected_models"
                ],
                target_column=target_column,
            )
        )

        # ========================================================
        # 8. MODEL EVALUATION
        # ========================================================

        print("\n[8/9] Model Evaluation")

        model_evaluation = ModelEvaluation(
            project_root=self.project_root
        )

        problem_type = (
            dataset_profile.get(
                "problem_type"
            )
        )

        if problem_type is None:

            raise ValueError(
                "Problem type was not found "
                "in dataset profile."
            )

        evaluation = (
            model_evaluation
            .initiate_model_evaluation(
                trained_models=model_training[
                    "trained_models"
                ],
                transformed_data=transformed_data,
                target_column=target_column,
                problem_type=problem_type,
            )
        )

        # ========================================================
        # 9. MODEL PUSHING & SAVING
        # ========================================================

        print("\n[9/9] Model Pushing & Saving")

        model_pusher = ModelPusher(
            project_root=self.project_root
        )

        # --------------------------------------------------------
        # Determine original input feature columns
        # --------------------------------------------------------
        #
        # These are the columns that the user will eventually
        # provide to the generic prediction UI.
        #
        # We use the cleaned training dataframe because identifier
        # columns such as "Id" may already have been removed.
        #

        feature_columns = [
            column
            for column in cleaned_train_data.columns
            if column != target_column
        ]

        # --------------------------------------------------------
        # Push best model
        # --------------------------------------------------------

        model_pushing = (
            model_pusher
            .initiate_model_pushing(

                best_model=evaluation[
                    "best_model"
                ],

                best_model_name=evaluation[
                    "best_model_name"
                ],

                best_model_score=evaluation[
                    "best_model_score"
                ],

                primary_metric=evaluation[
                    "primary_metric"
                ],

                problem_type=problem_type,

                evaluation_source=evaluation[
                    "evaluation_source"
                ],

                evaluation_samples=evaluation[
                    "evaluation_samples"
                ],

                evaluation_report=evaluation[
                    "evaluation_report"
                ],

                target_column=target_column,

                feature_columns=feature_columns,
            )
        )

        # ========================================================
        # PIPELINE SUMMARY
        # ========================================================

        print("\n" + "=" * 60)
        print("PIPELINE STATUS")
        print("=" * 60)

        print("Data Ingestion      : COMPLETED")
        print("Data Validation     : COMPLETED")
        print("Dataset Profiling   : COMPLETED")
        print("Data Cleaning       : COMPLETED")
        print("Data Transformation : COMPLETED")
        print("Model Selection     : COMPLETED")
        print("Model Training      : COMPLETED")
        print("Model Evaluation    : COMPLETED")
        print("Model Pushing       : COMPLETED")

        print(
            "\nTraining pipeline completed successfully."
        )

        # ========================================================
        # RETURN RESULTS
        # ========================================================

        return {

            "train_data":
                train_data,

            "validation_data":
                validation_data,

            "test_data":
                test_data,

            "target_column":
                target_column,

            "validation_report":
                validation_report,

            "dataset_profile":
                dataset_profile,

            "cleaned_train_data":
                cleaned_train_data,

            "cleaned_validation_data":
                cleaned_validation_data,

            "cleaned_test_data":
                cleaned_test_data,

            "cleaning_report":
                cleaning_report,

            "transformed_data":
                transformed_data,

            "model_selection":
                model_selection,

            "model_training":
                model_training,

            "model_evaluation":
                evaluation,

            "model_pushing":
                model_pushing,
        }

    # ============================================================
    # Target Column Detection
    # ============================================================

    def _detect_target_column(
        self,
        train_data: pd.DataFrame,
    ) -> str:
        """
        Detect the most likely target column.
        """

        if train_data is None:

            raise ValueError(
                "Training dataset is None."
            )

        if train_data.empty:

            raise ValueError(
                "Training dataset is empty."
            )

        columns = list(
            train_data.columns
        )

        if len(columns) < 2:

            raise ValueError(
                "Target detection requires "
                "at least two columns."
            )

        metadata_target = (
            train_data.attrs.get(
                "target_column"
            )
        )

        if (
            metadata_target
            and metadata_target in columns
        ):

            return metadata_target

        candidates = (
            self._build_target_candidates(
                train_data
            )
        )

        if not candidates:

            raise ValueError(
                "Unable to generate target "
                "column candidates."
            )

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        best_column = candidates[0][0]
        best_score = candidates[0][1]

        second_score = (
            candidates[1][1]
            if len(candidates) > 1
            else 0.0
        )

        score_gap = (
            best_score - second_score
        )

        if (
            best_score >= 70
            and score_gap >= 15
        ):

            return best_column

        if (
            best_score >= 60
            and score_gap >= 20
        ):

            return best_column

        print(
            "\nTarget column could not be "
            "determined automatically."
        )

        print(
            "Recommended target columns:"
        )

        display_count = min(
            5,
            len(candidates),
        )

        for index in range(
            display_count
        ):

            column, score, _ = (
                candidates[index]
            )

            print(
                f"  {index + 1}. "
                f"{column} "
                f"(score: {score:.1f})"
            )

        print(
            "\nArchForge will not guess "
            "an ambiguous target."
        )

        while True:

            choice = input(
                "\nSelect target column "
                "(number or exact name): "
            ).strip()

            if choice.isdigit():

                index = int(choice)

                if (
                    1
                    <= index
                    <= display_count
                ):

                    selected = (
                        candidates[
                            index - 1
                        ][0]
                    )

                    print(
                        f"Target selected: "
                        f"{selected}"
                    )

                    return selected

            if choice in columns:

                print(
                    f"Target selected: "
                    f"{choice}"
                )

                return choice

            print(
                "Invalid selection."
            )

    # ============================================================
    # Build Target Candidates
    # ============================================================

    def _build_target_candidates(
        self,
        data: pd.DataFrame,
    ) -> List[
        Tuple[str, float, str]
    ]:
        """
        Score every column as a possible target.
        """

        candidates = []

        total_rows = len(data)

        for column in data.columns:

            series = data[column]

            score = 0.0
            reasons = []

            if self._looks_like_identifier(
                series,
                column,
            ):

                score -= 80

                reasons.append(
                    "identifier-like"
                )

            name_score = (
                self._target_name_score(
                    column
                )
            )

            score += name_score

            if name_score > 0:

                reasons.append(
                    "target-like name"
                )

            if pd.api.types.is_numeric_dtype(
                series
            ):

                score += 15

                reasons.append(
                    "numeric"
                )

                unique_count = (
                    series.nunique(
                        dropna=True
                    )
                )

                unique_ratio = (
                    unique_count
                    / total_rows
                    if total_rows > 0
                    else 0.0
                )

                if unique_ratio >= 0.30:

                    score += 25

                    reasons.append(
                        "continuous values"
                    )

                elif unique_ratio >= 0.10:

                    score += 10

                elif (
                    unique_count >= 2
                    and unique_count <= 20
                ):

                    score += 12

                    reasons.append(
                        "discrete values"
                    )

                if (
                    unique_ratio >= 0.95
                    and total_rows > 20
                ):

                    score += 5

                    reasons.append(
                        "high target uniqueness"
                    )

            else:

                unique_count = (
                    series.nunique(
                        dropna=True
                    )
                )

                unique_ratio = (
                    unique_count
                    / total_rows
                    if total_rows > 0
                    else 0.0
                )

                if (
                    2
                    <= unique_count
                    <= 20
                    and unique_ratio <= 0.20
                ):

                    score += 30

                    reasons.append(
                        "class-like values"
                    )

                elif (
                    unique_count <= 50
                    and unique_ratio <= 0.30
                ):

                    score += 12

                    reasons.append(
                        "categorical values"
                    )

                elif unique_ratio > 0.50:

                    score -= 25

                    reasons.append(
                        "high-cardinality"
                    )

            missing_ratio = (
                series.isnull().mean()
            )

            if missing_ratio == 0:

                score += 5

            elif missing_ratio <= 0.05:

                score += 2

            elif missing_ratio <= 0.20:

                score -= 5

                reasons.append(
                    "some missing values"
                )

            else:

                score -= 25

                reasons.append(
                    "many missing values"
                )

            if (
                series.nunique(
                    dropna=True
                )
                <= 1
            ):

                score -= 60

                reasons.append(
                    "constant column"
                )

            if (
                series.notna().sum()
                == 0
            ):

                score -= 80

                reasons.append(
                    "empty column"
                )

            reason = (
                ", ".join(reasons)
                if reasons
                else "general candidate"
            )

            candidates.append(
                (
                    column,
                    max(
                        0.0,
                        score,
                    ),
                    reason,
                )
            )

        return candidates

    # ============================================================
    # Target Name Scoring
    # ============================================================

    @staticmethod
    def _target_name_score(
        column: str,
    ) -> float:

        name = str(
            column
        ).strip().lower()

        strong_names = {

            "target",
            "label",
            "output",
            "price",
            "cost",
            "salary",
            "profit",
            "revenue",
            "value",
            "score",
            "rating",
            "class",
            "result",
            "response",
            "prediction",
            "dependent",
            "dependent_variable",
            "y_true",
        }

        if name in strong_names:

            return 30.0

        strong_patterns = (

            "target",
            "label",
            "price",
            "cost",
            "salary",
            "profit",
            "revenue",
            "output",
            "result",
            "response",
            "prediction",
            "dependent",
        )

        for pattern in strong_patterns:

            if pattern in name:

                return 22.0

        if name in {
            "y",
            "z",
            "t",
            "p",
            "pr",
        }:

            return 10.0

        return 0.0

    # ============================================================
    # Identifier Detection
    # ============================================================

    @staticmethod
    def _looks_like_identifier(
        series: pd.Series,
        column: str,
    ) -> bool:

        name = str(
            column
        ).strip().lower()

        identifier_names = {

            "id",
            "index",
            "uuid",
            "identifier",
            "rowid",
            "row_id",
            "record_id",
            "recordid",
            "serial",
        }

        if name in identifier_names:

            return True

        identifier_patterns = (

            "_id",
            "id_",
            "identifier",
            "uuid",
            "row_number",
            "row_no",
            "record_number",
            "record_no",
            "customer_number",
            "transaction_number",
            "serial_number",
        )

        for pattern in identifier_patterns:

            if pattern in name:

                return True

        non_null = series.dropna()

        if non_null.empty:

            return False

        if len(non_null) <= 20:

            return False

        if not pd.api.types.is_integer_dtype(
            non_null
        ):

            return False

        unique_ratio = (
            non_null.nunique()
            / len(non_null)
        )

        if unique_ratio < 0.98:

            return False

        values = non_null.to_numpy()

        differences = np.diff(
            np.sort(values)
        )

        if (
            len(differences) > 0
            and np.all(
                differences == 1
            )
        ):

            return True

        return False

    # ============================================================
    # Public Helper
    # ============================================================

    def get_target_column(
        self,
        train_data: pd.DataFrame,
    ) -> str:

        return self._detect_target_column(
            train_data
        )
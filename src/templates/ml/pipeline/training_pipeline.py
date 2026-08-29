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
9. Model Saving

Target detection is handled internally by this pipeline.
No separate target_detector.py file is required.

Target Detection
----------------
ArchForge does NOT assume that the target column is the
last column.

Every column is evaluated using multiple signals:

    • Target-like column name
    • Data type
    • Number of unique values
    • Continuous/discrete behavior
    • Missing-value ratio
    • Identifier-like behavior
    • Dataset size

If confidence is high:
    -> target is selected automatically.

If confidence is low:
    -> ArchForge displays recommended candidates
    -> user selects the target column.

This prevents ArchForge from blindly selecting the
last column of an arbitrary dataset.
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

        print(
            f"\nProject Root: "
            f"{self.project_root}"
        )

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
            data_validation
            .initiate_data_validation(
                train_data=train_data,
                validation_data=validation_data,
                test_data=test_data,
            )
        )

        # ========================================================
        # 3. TARGET DETECTION + DATA PROFILING
        # ========================================================

        print("\n[3/9] Dataset Profiling")

        target_column = (
            self._detect_target_column(
                train_data
            )
        )

        print(
            f"Target column: "
            f"{target_column}"
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
        ) = (
            data_cleaning
            .initiate_data_cleaning(
                train_data=train_data,
                validation_data=validation_data,
                test_data=test_data,
            )
        )

        # ========================================================
        # 5. DATA TRANSFORMATION
        # ========================================================

        print("\n[5/9] Data Transformation")

        data_transformation = (
            DataTransformation(
                project_root=self.project_root
            )
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
            model_selector
            .select_models(
                dataset_profile=dataset_profile
            )
        )

        # ========================================================
        # 7-9 STATUS
        # ========================================================

        print(
            "\n[7/9] Model Training : PENDING"
        )

        print(
            "[8/9] Model Evaluation : PENDING"
        )

        print(
            "[9/9] Model Saving : PENDING"
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "CURRENT PIPELINE STATUS"
        )

        print(
            "=" * 60
        )

        print(
            "[1/9] Data Ingestion      : COMPLETED"
        )

        print(
            "[2/9] Data Validation     : COMPLETED"
        )

        print(
            "[3/9] Dataset Profiling   : COMPLETED"
        )

        print(
            "[4/9] Data Cleaning       : COMPLETED"
        )

        print(
            "[5/9] Data Transformation : COMPLETED"
        )

        print(
            "[6/9] Model Selection     : COMPLETED"
        )

        print(
            "[7/9] Model Training      : PENDING"
        )

        print(
            "[8/9] Model Evaluation    : PENDING"
        )

        print(
            "[9/9] Model Saving        : PENDING"
        )

        print(
            "\nPipeline stages completed successfully."
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

        The last column is NEVER automatically assumed
        to be the target.

        Detection uses multiple independent signals.

        High confidence:
            Automatically select target.

        Low confidence:
            Display recommended candidates and ask
            the user to select one.
        """

        # --------------------------------------------------------
        # Basic validation
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Data ingestion metadata has highest priority
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Build candidates
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Sort by score
        # --------------------------------------------------------

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
            best_score
            - second_score
        )

        # ========================================================
        # HIGH CONFIDENCE
        # ========================================================

        if (
            best_score >= 70
            and score_gap >= 15
        ):

            return best_column

        # ========================================================
        # MODERATE-HIGH CONFIDENCE
        # ========================================================

        if (
            best_score >= 60
            and score_gap >= 20
        ):

            return best_column

        # ========================================================
        # LOW CONFIDENCE
        # ========================================================

        print(
            "\nTarget column could not be "
            "determined with high confidence."
        )

        print(
            "\nRecommended target columns:"
        )

        display_count = min(
            5,
            len(candidates),
        )

        for index in range(
            display_count
        ):

            (
                column,
                score,
                reason,
            ) = candidates[index]

            print(
                f"  {index + 1}. "
                f"{column} "
                f"(score: {score:.1f})"
            )

        print(
            "\nArchForge will not automatically "
            "guess the target."
        )

        # ========================================================
        # USER SELECTION
        # ========================================================

        while True:

            choice = input(
                "\nSelect target column "
                "(number or exact column name): "
            ).strip()

            # ----------------------------------------------------
            # Number selection
            # ----------------------------------------------------

            if choice.isdigit():

                index = int(
                    choice
                )

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
                        f"\nTarget selected: "
                        f"{selected}"
                    )

                    return selected

            # ----------------------------------------------------
            # Exact column selection
            # ----------------------------------------------------

            if choice in columns:

                print(
                    f"\nTarget selected: "
                    f"{choice}"
                )

                return choice

            print(
                "Invalid selection. "
                "Please enter a valid number "
                "or column name."
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

        Target scoring combines:

            1. Column name
            2. Data type
            3. Unique-value behavior
            4. Continuous/discrete behavior
            5. Missing values
            6. Identifier behavior
            7. Dataset size

        No single signal is sufficient by itself.
        """

        candidates = []

        total_rows = len(
            data
        )

        for column in data.columns:

            series = data[
                column
            ]

            score = 0.0
            reasons = []

            # ====================================================
            # 1. Identifier Detection
            # ====================================================

            if self._looks_like_identifier(
                series,
                column,
            ):

                score -= 80

                reasons.append(
                    "identifier-like"
                )

            # ====================================================
            # 2. Target Name Signal
            # ====================================================

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

            # ====================================================
            # 3. Numeric Features
            # ====================================================

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

                # ------------------------------------------------
                # Continuous numeric behavior
                # ------------------------------------------------

                if unique_ratio >= 0.30:

                    score += 25

                    reasons.append(
                        "continuous values"
                    )

                elif unique_ratio >= 0.10:

                    score += 10

                    reasons.append(
                        "moderate uniqueness"
                    )

                # ------------------------------------------------
                # Very low-cardinality numerical target
                # ------------------------------------------------

                elif (
                    unique_count >= 2
                    and unique_count <= 20
                ):

                    score += 12

                    reasons.append(
                        "discrete values"
                    )

                # ------------------------------------------------
                # Almost every row unique
                #
                # This can be a valid regression target,
                # so do NOT automatically reject it.
                # ------------------------------------------------

                if (
                    unique_ratio >= 0.95
                    and total_rows > 20
                ):

                    score += 5

                    reasons.append(
                        "high target uniqueness"
                    )

            # ====================================================
            # 4. Categorical Features
            # ====================================================

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

                # ------------------------------------------------
                # Classification-like target
                # ------------------------------------------------

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

                # ------------------------------------------------
                # Moderate categorical cardinality
                # ------------------------------------------------

                elif (
                    unique_count <= 50
                    and unique_ratio <= 0.30
                ):

                    score += 12

                    reasons.append(
                        "categorical values"
                    )

                # ------------------------------------------------
                # High-cardinality text
                # ------------------------------------------------

                elif unique_ratio > 0.50:

                    score -= 25

                    reasons.append(
                        "high-cardinality"
                    )

            # ====================================================
            # 5. Missing Values
            # ====================================================

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

            # ====================================================
            # 6. Constant Column
            # ====================================================

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

            # ====================================================
            # 7. Empty Column
            # ====================================================

            if (
                series.notna().sum()
                == 0
            ):

                score -= 80

                reasons.append(
                    "empty column"
                )

            # ====================================================
            # 8. Store Candidate
            # ====================================================

            reason = (
                ", ".join(
                    reasons
                )
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
        """
        Score the column name as a weak target signal.

        IMPORTANT:
        The name alone never determines the target.

        Examples:

            Price
            Cost
            Salary
            Revenue
            Target
            Output
            y
            pr

        are considered, but data characteristics
        remain important.
        """

        name = str(
            column
        ).strip().lower()

        # --------------------------------------------------------
        # Strong exact names
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Common target patterns
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Short mathematical names
        #
        # These get a SMALL bonus because:
        #
        # y -> could be target
        # x -> could be feature
        # p -> could be price/probability/etc.
        #
        # We should NOT heavily trust them.
        # --------------------------------------------------------

        short_target_names = {
            "y",
            "z",
            "t",
            "p",
            "pr",
        }

        if name in short_target_names:

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
        """
        Detect whether a column behaves like an identifier.

        Examples:

            Id
            ID
            customer_id
            transaction_id
            row_number
            uuid

        Identifier detection uses both:

            • column name
            • data behavior

        A continuous numerical target with mostly unique
        values is NOT automatically classified as an ID.
        """

        name = str(
            column
        ).strip().lower()

        # ========================================================
        # Explicit identifier names
        # ========================================================

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

        # ========================================================
        # Identifier name patterns
        # ========================================================

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

        # ========================================================
        # Data-behavior detection
        # ========================================================

        if len(series) == 0:

            return False

        non_null = (
            series.dropna()
        )

        if non_null.empty:

            return False

        unique_count = (
            non_null.nunique()
        )

        row_count = len(
            non_null
        )

        if row_count <= 20:

            return False

        unique_ratio = (
            unique_count
            / row_count
        )

        # ========================================================
        # Sequential integer identifier
        # ========================================================

        if pd.api.types.is_integer_dtype(
            non_null
        ):

            values = (
                non_null
                .to_numpy()
            )

            sorted_values = np.sort(
                values
            )

            differences = np.diff(
                sorted_values
            )

            if (
                unique_ratio >= 0.98
                and len(differences) > 0
                and np.all(
                    differences == 1
                )
            ):

                return True

        # ========================================================
        # Very high uniqueness alone is NOT enough.
        #
        # Example:
        #
        # House Price:
        # 1599 unique / 1600 rows
        #
        # This is a perfectly valid regression target.
        #
        # Therefore we intentionally return False here.
        # ========================================================

        return False

    # ============================================================
    # Public Helper
    # ============================================================

    def get_target_column(
        self,
        train_data: pd.DataFrame,
    ) -> str:
        """
        Public wrapper for target detection.
        """

        return self._detect_target_column(
            train_data
        )
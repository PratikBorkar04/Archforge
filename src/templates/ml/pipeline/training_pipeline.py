"""
training_pipeline.py

Main orchestration pipeline for ArchForge ML projects.

Pipeline
--------
1. Data Ingestion
2. Data Validation
3. Data Profiling
4. Data Cleaning
5. Data Transformation
6. Model Training
7. Model Evaluation
8. Model Saving

The user only needs to run:

    python app.py

All pipeline stages are executed automatically.
"""

from pathlib import Path
from typing import Dict, Any

from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation
from src.dataset_profiler import DatasetProfiler
from src.data_cleaning import DataCleaning
from src.data_transformation import DataTransformation


class TrainingPipeline:
    """
    Orchestrates the complete ML training workflow.

    The pipeline is designed so that the user does not need
    to manually execute individual ML components.
    """

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

        Currently implemented stages:

            1. Data Ingestion
            2. Data Validation
            3. Dataset Profiling
            4. Data Cleaning
            5. Data Transformation

        Model training, evaluation and saving will be
        implemented in the next stages.
        """

        print(
            "\n"
            + "=" * 60
        )

        print(
            "Starting Machine Learning Pipeline"
        )

        print(
            "=" * 60
        )

        print(
            f"\nProject Root: "
            f"{self.project_root}"
        )

        # ========================================================
        # 1. DATA INGESTION
        # ========================================================

        print(
            "\n[1/8] Data Ingestion"
        )

        data_ingestion = DataIngestion(
            project_root=self.project_root
        )

        (
            train_data,
            validation_data,
            test_data,
        ) = (
            data_ingestion
            .initiate_data_ingestion()
        )

        # ========================================================
        # 2. DATA VALIDATION
        # ========================================================

        print(
            "\n[2/8] Data Validation"
        )

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
        # 3. DATA PROFILING
        # ========================================================

        print(
            "\n[3/8] Dataset Profiling"
        )

        # --------------------------------------------------------
        # Determine target column
        # --------------------------------------------------------
        # Data ingestion currently guarantees that the target
        # column is available as metadata or as the final column.
        # --------------------------------------------------------

        target_column = (
            self._get_target_column(
                train_data
            )
        )

        print(
            f"\nTarget column: "
            f"{target_column}"
        )

        # --------------------------------------------------------
        # Profile TRAINING data only
        # --------------------------------------------------------
        #
        # We intentionally do not profile the test dataset.
        #
        # The test dataset must remain unseen for final evaluation.
        #
        # Profiling the training data allows the model selector
        # to make decisions based on the available training data.
        # --------------------------------------------------------

        profiler = DatasetProfiler(
            project_root=self.project_root,
            target_column=target_column,
        )

        dataset_profile = (
            profiler.profile(
                train_data
            )
        )

        # ========================================================
        # 4. DATA CLEANING
        # ========================================================

        print(
            "\n[4/8] Data Cleaning"
        )

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
        # IMPORTANT
        # ========================================================
        #
        # From this point onward we MUST use the cleaned datasets.
        #
        # Do not accidentally pass the original train_data,
        # validation_data or test_data to later stages.
        # ========================================================

        # ========================================================
        # 5. DATA TRANSFORMATION
        # ========================================================

        print(
            "\n[5/8] Data Transformation"
        )

        data_transformation = (
            DataTransformation(
                project_root=self.project_root
            )
        )

        transformed_data = (
            data_transformation
            .initiate_data_transformation(
                train_data=(
                    cleaned_train_data
                ),
                validation_data=(
                    cleaned_validation_data
                ),
                test_data=(
                    cleaned_test_data
                ),
            )
        )

        # ========================================================
        # TEMPORARY SUMMARY
        # ========================================================

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
            "\n[1/8] Data Ingestion      : COMPLETED"
        )

        print(
            "[2/8] Data Validation     : COMPLETED"
        )

        print(
            "[3/8] Dataset Profiling   : COMPLETED"
        )

        print(
            "[4/8] Data Cleaning       : COMPLETED"
        )

        print(
            "[5/8] Data Transformation : COMPLETED"
        )

        print(
            "[6/8] Model Training      : PENDING"
        )

        print(
            "[7/8] Model Evaluation    : PENDING"
        )

        print(
            "[8/8] Model Saving        : PENDING"
        )

        print(
            "\nPipeline stages completed successfully."
        )

        # ========================================================
        # Return pipeline outputs
        # ========================================================

        return {

            # ----------------------------------------------------
            # Original data
            # ----------------------------------------------------

            "train_data":
                train_data,

            "validation_data":
                validation_data,

            "test_data":
                test_data,

            # ----------------------------------------------------
            # Validation
            # ----------------------------------------------------

            "validation_report":
                validation_report,

            # ----------------------------------------------------
            # Profiling
            # ----------------------------------------------------

            "dataset_profile":
                dataset_profile,

            # ----------------------------------------------------
            # Cleaned data
            # ----------------------------------------------------

            "cleaned_train_data":
                cleaned_train_data,

            "cleaned_validation_data":
                cleaned_validation_data,

            "cleaned_test_data":
                cleaned_test_data,

            "cleaning_report":
                cleaning_report,

            # ----------------------------------------------------
            # Transformation
            # ----------------------------------------------------

            "transformed_data":
                transformed_data,
        }

    # ============================================================
    # Target Column Detection
    # ============================================================

    @staticmethod
    def _get_target_column(
        train_data,
    ) -> str:
        """
        Determine the target column from the training dataset.

        Preferred method:
            Read target metadata attached by Data Ingestion.

        Fallback:
            Use the final column.

        The fallback exists because the current Data Ingestion
        implementation guarantees that the target is the final
        column when explicit metadata is unavailable.
        """

        # --------------------------------------------------------
        # 1. Check DataFrame metadata
        # --------------------------------------------------------

        if hasattr(
            train_data,
            "attrs",
        ):

            target_column = (
                train_data.attrs.get(
                    "target_column"
                )
            )

            if target_column:

                return target_column

        # --------------------------------------------------------
        # 2. Validate dataset
        # --------------------------------------------------------

        if train_data is None:

            raise ValueError(
                "Training dataset is None."
            )

        if len(
            train_data.columns
        ) == 0:

            raise ValueError(
                "Training dataset contains "
                "no columns."
            )

        # --------------------------------------------------------
        # 3. Temporary fallback
        # --------------------------------------------------------
        #
        # Data ingestion currently places the target column
        # at the end of the dataset.
        # --------------------------------------------------------

        return train_data.columns[-1]
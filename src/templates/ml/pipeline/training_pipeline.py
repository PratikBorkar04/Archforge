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
6. Model Selection
7. Model Training
8. Model Evaluation
9. Model Saving

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
from src.model_selector import ModelSelector


class TrainingPipeline:
    """
    Orchestrates the complete ML training workflow.

    The pipeline automatically executes each stage and passes
    the output of one stage to the next stage.
    """

    def __init__(
        self,
        project_root: Path,
    ) -> None:

        self.project_root = Path(
            project_root
        ).resolve()

    # ============================================================
    # RUN PIPELINE
    # ============================================================

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete ML pipeline.

        Current implemented stages:

            1. Data Ingestion
            2. Data Validation
            3. Dataset Profiling
            4. Data Cleaning
            5. Data Transformation
            6. Model Selection

        Model training, evaluation and saving will be implemented
        after the model-selection stage has been tested.
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
            "\n[1/9] Data Ingestion"
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
            "\n[2/9] Data Validation"
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
            "\n[3/9] Dataset Profiling"
        )

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
        # Profile training data only.
        #
        # Test data must remain unseen until final evaluation.
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
            "\n[4/9] Data Cleaning"
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
        # From this point onward, only cleaned datasets are used.
        # ========================================================

        # ========================================================
        # 5. DATA TRANSFORMATION
        # ========================================================

        print(
            "\n[5/9] Data Transformation"
        )

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

        print(
            "\n[6/9] Model Selection"
        )

        model_selector = ModelSelector(
            project_root=self.project_root
        )

        # --------------------------------------------------------
        # IMPORTANT
        # --------------------------------------------------------
        #
        # The current ModelSelector interface accepts:
        #
        #     dataset_profile
        #
        # We therefore DO NOT pass transformed_data or
        # target_column here.
        #
        # Model selection decides which algorithms are suitable
        # for the dataset.
        #
        # It does NOT train the models.
        # --------------------------------------------------------

        model_selection = (
            model_selector
            .select_models(
                dataset_profile=dataset_profile
            )
        )

        # ========================================================
        # 7. MODEL TRAINING
        # ========================================================

        print(
            "\n[7/9] Model Training : PENDING"
        )

        # --------------------------------------------------------
        # ModelTraining will be added after ModelSelector has
        # been completely tested.
        # --------------------------------------------------------

        training_result = None

        # ========================================================
        # 8. MODEL EVALUATION
        # ========================================================

        print(
            "\n[8/9] Model Evaluation : PENDING"
        )

        # --------------------------------------------------------
        # Evaluation will compare trained models using metrics
        # appropriate for regression/classification.
        # --------------------------------------------------------

        evaluation_result = None

        # ========================================================
        # 9. MODEL SAVING
        # ========================================================

        print(
            "\n[9/9] Model Saving : PENDING"
        )

        # --------------------------------------------------------
        # Model saving will be implemented after training and
        # evaluation.
        # --------------------------------------------------------

        saving_result = None

        # ========================================================
        # PIPELINE SUMMARY
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
            "\n[1/9] Data Ingestion      : COMPLETED"
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
            "\nModel selection stage completed."
        )

        # ========================================================
        # RETURN PIPELINE OUTPUTS
        # ========================================================

        return {

            # ----------------------------------------------------
            # Original datasets
            # ----------------------------------------------------

            "train_data":
                train_data,

            "validation_data":
                validation_data,

            "test_data":
                test_data,

            # ----------------------------------------------------
            # Target
            # ----------------------------------------------------

            "target_column":
                target_column,

            # ----------------------------------------------------
            # Validation
            # ----------------------------------------------------

            "validation_report":
                validation_report,

            # ----------------------------------------------------
            # Dataset profile
            # ----------------------------------------------------

            "dataset_profile":
                dataset_profile,

            # ----------------------------------------------------
            # Cleaned datasets
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

            # ----------------------------------------------------
            # Model selection
            # ----------------------------------------------------

            "model_selection":
                model_selection,

            # ----------------------------------------------------
            # Future stages
            # ----------------------------------------------------

            "training_result":
                training_result,

            "evaluation_result":
                evaluation_result,

            "saving_result":
                saving_result,
        }

    # ============================================================
    # TARGET COLUMN DETECTION
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

        Data Ingestion currently guarantees that the target
        column is placed at the end when explicit metadata is
        unavailable.
        """

        # --------------------------------------------------------
        # Validate dataset
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
        # Check DataFrame metadata
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
        # Fallback
        # --------------------------------------------------------

        return train_data.columns[-1]

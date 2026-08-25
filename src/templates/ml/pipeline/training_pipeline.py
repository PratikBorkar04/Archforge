"""
training_pipeline.py

Main orchestration pipeline for ArchForge ML projects.

Pipeline
--------
1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation
6. Model Saving

The user only needs to run:

    python app.py

All pipeline stages are executed automatically.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation
from src.data_transformation import DataTransformation


class TrainingPipeline:
    """
    Orchestrates the complete ML training workflow.
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

            Data Ingestion
            Data Validation
            Data Transformation

        Model training, evaluation and saving
        will be added in the next stages.
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
        # 3. DATA TRANSFORMATION
        # ========================================================

        data_transformation = (
            DataTransformation(
                project_root=self.project_root
            )
        )

        transformed_data = (
            data_transformation
            .initiate_data_transformation(
                train_data=train_data,
                validation_data=validation_data,
                test_data=test_data,
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
            "\n[1/6] Data Ingestion      : COMPLETED"
        )

        print(
            "[2/6] Data Validation     : COMPLETED"
        )

        print(
            "[3/6] Data Transformation : COMPLETED"
        )

        print(
            "[4/6] Model Training      : PENDING"
        )

        print(
            "[5/6] Model Evaluation    : PENDING"
        )

        print(
            "[6/6] Model Saving        : PENDING"
        )

        print(
            "\nPipeline stages completed successfully."
        )

        # ========================================================
        # Return pipeline outputs
        # ========================================================

        return {
            "train_data": train_data,

            "validation_data":
                validation_data,

            "test_data": test_data,

            "validation_report":
                validation_report,

            "transformed_data":
                transformed_data,
        }
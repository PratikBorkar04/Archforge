"""
Training Pipeline
-----------------

Coordinates the complete machine learning workflow.

The user only needs to run:

    python app.py

The pipeline automatically executes each stage.
"""

from pathlib import Path

from src.data_ingestion import DataIngestion


class TrainingPipeline:
    """
    Coordinates the complete machine learning training workflow.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()

    def run(self) -> None:
        """
        Execute the complete machine learning pipeline.
        """

        print("\n" + "=" * 60)
        print("Starting Machine Learning Pipeline")
        print("=" * 60)

        print(f"\nProject Root: {self.project_root}")

        # ----------------------------------------------------------
        # 1. Data Ingestion
        # ----------------------------------------------------------

        data_ingestion = DataIngestion(
            project_root=self.project_root
        )

        train_data, test_data = (
            data_ingestion.initiate_data_ingestion()
        )

        print("\nData ingestion completed successfully.")

        print(
            f"Training samples : {len(train_data)}"
        )

        print(
            f"Testing samples  : {len(test_data)}"
        )

        print("\nPipeline stage completed.")
"""
Training Pipeline
-----------------

Coordinates the complete machine learning training workflow.
"""

from pathlib import Path


class TrainingPipeline:
    """
    Coordinates the complete machine learning training workflow.

    Parameters
    ----------
    project_root : Path
        Root directory of the generated ML project.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()

    def run(self) -> None:
        """Execute the machine learning training pipeline."""

        print("\n" + "=" * 60)
        print("Starting Machine Learning Pipeline")
        print("=" * 60)

        print(f"\nProject Root: {self.project_root}")

        print("\nPipeline initialized successfully.")
"""
Application Entry Point
-----------------------

Main entry point of the generated ML project.

Usage
-----
Run the complete machine learning pipeline with:

    python app.py

The application automatically executes:

    Data Ingestion
        ↓
    Data Validation
        ↓
    Data Transformation
        ↓
    Model Training
        ↓
    Model Evaluation
        ↓
    Model Saving

The user does not need to run individual components manually.
"""

from pathlib import Path
import sys


# ------------------------------------------------------------------
# Project Root
# ------------------------------------------------------------------
# Resolve the directory containing this app.py file.
# This makes the project independent of the current terminal path
# and keeps file handling compatible with Windows and Linux.

PROJECT_ROOT = Path(__file__).resolve().parent


# ------------------------------------------------------------------
# Python Import Path
# ------------------------------------------------------------------
# Add the generated project's root directory to Python's import path.
# insert(0, ...) gives our project priority over other directories.

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Import after configuring the project path.
from pipeline.training_pipeline import TrainingPipeline


def main() -> None:
    """
    Start the complete machine learning training pipeline.
    """

    pipeline = TrainingPipeline(
        project_root=PROJECT_ROOT
    )

    pipeline.run()


if __name__ == "__main__":
    main()
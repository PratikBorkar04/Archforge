"""
Data Ingestion
--------------

Automatically prepares the dataset for the ML pipeline.

Supported behavior:

    1 CSV
        → use directly

    Multiple CSVs with same schema
        → concatenate

    Multiple related CSVs
        → identify possible relationship

    train.csv / test.csv / validation.csv
        → keep separate

    Unrelated datasets
        → ask the user instead of merging blindly
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_analysis import DataAnalyzer


class DataIngestion:
    """
    Automatically discover and prepare datasets.
    """

    def __init__(
        self,
        project_root: Path,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:

        self.project_root = Path(
            project_root
        ).resolve()

        self.raw_data_dir = (
            self.project_root / "data" / "raw"
        )

        self.artifacts_dir = (
            self.project_root / "artifacts"
        )

        self.test_size = test_size
        self.random_state = random_state

    # --------------------------------------------------------------
    # Combine same-schema datasets
    # --------------------------------------------------------------

    def _combine_same_schema_files(
        self,
        files,
    ) -> pd.DataFrame:
        """
        Concatenate datasets that have the same schema.
        """

        datasets = []

        for file in files:

            data = pd.read_csv(file)

            print(
                f"Loading: {file.name} "
                f"({len(data)} rows)"
            )

            datasets.append(data)

        if not datasets:
            raise ValueError(
                "No datasets available for combination."
            )

        combined = pd.concat(
            datasets,
            ignore_index=True,
        )

        print(
            f"\nCombined dataset: "
            f"{len(combined)} rows × "
            f"{len(combined.columns)} columns"
        )

        return combined

    # --------------------------------------------------------------
    # Ask user about unrelated datasets
    # --------------------------------------------------------------

    @staticmethod
    def _ask_dataset_choice(
        dataset_groups,
    ):
        """
        Ask the user which dataset group should be used.
        """

        print("\n" + "=" * 60)
        print("MULTIPLE DATASETS DETECTED")
        print("=" * 60)

        print(
            "\nArchForge detected multiple "
            "independent datasets."
        )

        for index, group in enumerate(
            dataset_groups,
            start=1,
        ):

            print(f"\n{index}. Dataset group")

            for file in group:
                print(f"   • {file.name}")

        print("\n0. Cancel")

        while True:

            choice = input(
                "\nSelect dataset to train on: "
            ).strip()

            if choice == "0":
                raise RuntimeError(
                    "Training cancelled by user."
                )

            try:
                selected = int(choice)

            except ValueError:
                print(
                    "Please enter a valid number."
                )
                continue

            if 1 <= selected <= len(
                dataset_groups
            ):

                return dataset_groups[
                    selected - 1
                ]

            print(
                "Invalid selection. "
                "Please choose one of the listed options."
            )

    # --------------------------------------------------------------
    # Prepare dataset
    # --------------------------------------------------------------

    def _prepare_dataset(
        self,
        analysis,
    ) -> pd.DataFrame:
        """
        Determine which dataset should be used.

        This method currently handles:
            - single dataset
            - same-schema datasets

        More advanced joining will be added separately.
        """

        other_files = analysis["other_files"]

        # ----------------------------------------------------------
        # Case 1: No ordinary dataset
        # ----------------------------------------------------------

        if not other_files:

            train_files = analysis["train_files"]

            if train_files:

                print(
                    "\nUsing explicitly provided "
                    "training dataset."
                )

                return pd.read_csv(
                    train_files[0]
                )

            raise ValueError(
                "No usable training dataset found."
            )

        # ----------------------------------------------------------
        # Case 2: One dataset
        # ----------------------------------------------------------

        if len(other_files) == 1:

            file = other_files[0]

            print(
                f"\nUsing dataset: {file.name}"
            )

            return pd.read_csv(file)

        # ----------------------------------------------------------
        # Case 3: Check relationships
        # ----------------------------------------------------------

        groups = []

        used_files = set()

        # First group files with same schema.
        for file in other_files:

            if file in used_files:
                continue

            group = [file]

            used_files.add(file)

            for other_file in other_files:

                if other_file in used_files:
                    continue

                relationship = (
                    DataAnalyzer(
                        self.project_root
                    ).analyze_relationship(
                        file,
                        other_file,
                    )
                )

                if relationship == "same_schema":

                    group.append(
                        other_file
                    )

                    used_files.add(
                        other_file
                    )

            groups.append(group)

        # ----------------------------------------------------------
        # Display discovered groups
        # ----------------------------------------------------------

        print(
            "\nDataset groups detected:"
        )

        for index, group in enumerate(
            groups,
            start=1,
        ):

            print(
                f"\nGroup {index}:"
            )

            for file in group:
                print(
                    f"  • {file.name}"
                )

        # ----------------------------------------------------------
        # If multiple independent groups exist,
        # ask the user instead of guessing.
        # ----------------------------------------------------------

        if len(groups) > 1:

            selected_group = (
                self._ask_dataset_choice(
                    groups
                )
            )

        else:

            selected_group = groups[0]

        # ----------------------------------------------------------
        # Combine selected group
        # ----------------------------------------------------------

        return self._combine_same_schema_files(
            selected_group
        )

    # --------------------------------------------------------------
    # Main ingestion
    # --------------------------------------------------------------

    def initiate_data_ingestion(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Automatically discover, prepare and split data.
        """

        print("\n[1/6] Data Ingestion")

        analyzer = DataAnalyzer(
            self.project_root
        )

        analysis = analyzer.analyze()

        # ----------------------------------------------------------
        # Prepare final dataset
        # ----------------------------------------------------------

        data = self._prepare_dataset(
            analysis
        )

        # ----------------------------------------------------------
        # Basic checks
        # ----------------------------------------------------------

        if data.empty:

            raise ValueError(
                "The selected dataset is empty."
            )

        if data.shape[1] < 2:

            raise ValueError(
                "Dataset must contain at least "
                "two columns."
            )

        # ----------------------------------------------------------
        # Target detection
        # ----------------------------------------------------------

        target_column = data.columns[-1]

        print(
            f"\nTarget column detected: "
            f"{target_column}"
        )

        # ----------------------------------------------------------
        # Train/test split
        # ----------------------------------------------------------

        train_data, test_data = (
            train_test_split(
                data,
                test_size=self.test_size,
                random_state=self.random_state,
            )
        )

        # ----------------------------------------------------------
        # Save artifacts
        # ----------------------------------------------------------

        self.artifacts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_path = (
            self.artifacts_dir / "train.csv"
        )

        test_path = (
            self.artifacts_dir / "test.csv"
        )

        train_data.to_csv(
            train_path,
            index=False,
        )

        test_data.to_csv(
            test_path,
            index=False,
        )

        print(
            f"\nTraining data saved: "
            f"{train_path}"
        )

        print(
            f"Testing data saved: "
            f"{test_path}"
        )

        print(
            "\nData ingestion completed successfully."
        )

        print(
            f"Training samples : "
            f"{len(train_data)}"
        )

        print(
            f"Testing samples  : "
            f"{len(test_data)}"
        )

        return train_data, test_data
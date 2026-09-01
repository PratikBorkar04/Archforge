"""
data_ingestion.py

Data ingestion module for ArchForge.

Responsibilities
----------------
1. Discover datasets from data/raw/.
2. Analyze multiple dataset files.
3. Combine files with the same schema.
4. Keep explicit train/test/validation datasets separate.
5. Detect the target column.
6. Automatically split normal datasets when required.
7. Save training, validation and testing data into artifacts/.

Important
---------
File order is NEVER used to determine train/test data.

Examples
--------

Normal dataset:

    house.csv
        ↓
    80% training
    20% testing

Multiple parts of the same dataset:

    house1.csv
    house2.csv
        ↓
    combine
        ↓
    80% training
    20% testing

Explicit datasets:

    house_train.csv
    house_validation.csv
    house_test.csv
        ↓
    DO NOT SPLIT
        ↓
    train / validation / test
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_analysis import DataAnalyzer


class DataIngestion:
    """
    Automatically discover and prepare datasets
    for the machine learning pipeline.
    """

    def __init__(
        self,
        project_root: Path,
        test_size: float = 0.20,
        validation_size: float = 0.20,
        random_state: int = 42,
    ) -> None:

        self.project_root = Path(
            project_root
        ).resolve()

        self.raw_data_dir = (
            self.project_root
            / "data"
            / "raw"
        )

        self.artifacts_dir = (
            self.project_root
            / "artifacts"
        )

        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

    # ============================================================
    # Read Dataset
    # ============================================================

    @staticmethod
    def _read_dataset(
        file: Path,
    ) -> pd.DataFrame:
        """
        Read a CSV dataset safely.

        Automatically detects the delimiter used
        by the dataset.

        Supported common delimiters include:

            ,
            ;
            \\t
            |
        """

        try:

            data = pd.read_csv(
                file,
                sep=None,
                engine="python",
            )

        except Exception as exc:

            raise RuntimeError(
                f"\nUnable to read dataset:\n"
                f"{file}\n\n"
                f"Reason: {exc}"
            ) from exc

        if data.empty:

            raise ValueError(
                f"\nDataset is empty:\n"
                f"{file.name}"
            )

        if len(data.columns) < 2:

            raise ValueError(
                f"\nDataset must contain at least "
                f"two columns.\n\n"
                f"File: {file.name}\n"
                f"Detected columns: "
                f"{list(data.columns)}"
            )

        return data

    # ============================================================
    # Combine Same-Schema Files
    # ============================================================

    def _combine_same_schema_files(
        self,
        files,
    ) -> pd.DataFrame:
        """
        Combine multiple CSV files having
        the same schema.

        Example:

            house1.csv
            house2.csv
            house3.csv

        become one dataset.
        """

        datasets = []

        print(
            "\nCombining datasets..."
        )

        for file in files:

            data = self._read_dataset(
                file
            )

            print(
                f"Loading: {file.name} "
                f"({len(data)} rows)"
            )

            datasets.append(data)

        if not datasets:

            raise ValueError(
                "No datasets available "
                "for combination."
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

    # ============================================================
    # Ask User Which Dataset to Use
    # ============================================================

    @staticmethod
    def _ask_dataset_choice(
        dataset_groups,
    ):
        """
        Ask the user which independent dataset
        should be used for the current project.
        """

        print(
            "\n"
            + "=" * 60
        )

        print(
            "MULTIPLE DATASETS DETECTED"
        )

        print(
            "=" * 60
        )

        print(
            "\nArchForge detected multiple "
            "independent datasets."
        )

        for index, group in enumerate(
            dataset_groups,
            start=1,
        ):

            print(
                f"\n{index}. Dataset group"
            )

            for file in group:

                print(
                    f"   • {file.name}"
                )

        print(
            "\n0. Cancel"
        )

        while True:

            choice = input(
                "\nSelect dataset to train on: "
            ).strip()

            if choice == "0":

                raise RuntimeError(
                    "Training cancelled by user."
                )

            try:

                selected = int(
                    choice
                )

            except ValueError:

                print(
                    "Please enter a valid number."
                )

                continue

            if (
                1
                <= selected
                <= len(dataset_groups)
            ):

                return dataset_groups[
                    selected - 1
                ]

            print(
                "Invalid selection. "
                "Please choose one of the "
                "listed options."
            )

    # ============================================================
    # Prepare Normal Dataset
    # ============================================================

    def _prepare_normal_dataset(
        self,
        analysis,
    ) -> pd.DataFrame:
        """
        Prepare a normal dataset when explicit
        train/test/validation files are not provided.

        Handles:

            1 CSV
                ↓
            use directly

            Multiple same-schema CSVs
                ↓
            combine

            Multiple unrelated datasets
                ↓
            ask user
        """

        other_files = analysis[
            "other_files"
        ]

        # --------------------------------------------------------
        # No normal dataset
        # --------------------------------------------------------

        if not other_files:

            raise ValueError(
                "\nNo normal training dataset found."
            )

        # --------------------------------------------------------
        # One dataset
        # --------------------------------------------------------

        if len(other_files) == 1:

            file = other_files[0]

            print(
                f"\nUsing dataset: "
                f"{file.name}"
            )

            return self._read_dataset(
                file
            )

        # --------------------------------------------------------
        # Dataset groups
        # --------------------------------------------------------

        groups = analysis[
            "dataset_groups"
        ]

        # --------------------------------------------------------
        # Multiple independent groups
        # --------------------------------------------------------

        if len(groups) > 1:

            selected_group = (
                self._ask_dataset_choice(
                    groups
                )
            )

        else:

            selected_group = groups[0]

        # --------------------------------------------------------
        # Combine selected group
        # --------------------------------------------------------

        return (
            self._combine_same_schema_files(
                selected_group
            )
        )

    # ============================================================
    # Prepare Explicit Datasets
    # ============================================================

    def _prepare_explicit_datasets(
        self,
        analysis,
    ) -> Tuple[
        pd.DataFrame,
        Optional[pd.DataFrame],
        pd.DataFrame,
    ]:
        """
        Prepare explicitly provided train,
        validation and test datasets.

        IMPORTANT
        ---------
        No random splitting is performed here.
        """

        train_files = analysis[
            "train_files"
        ]

        validation_files = analysis[
            "validation_files"
        ]

        test_files = analysis[
            "test_files"
        ]

        # --------------------------------------------------------
        # Training dataset
        # --------------------------------------------------------

        if not train_files:

            raise ValueError(
                "\nA training dataset was detected "
                "as an explicit split, but no "
                "training file was found."
            )

        if len(train_files) > 1:

            raise ValueError(
                "\nMultiple training datasets detected:\n"
                + "\n".join(
                    f"  • {file.name}"
                    for file in train_files
                )
                + "\n\n"
                "Please provide a single training "
                "dataset or combine the files first."
            )

        train_file = train_files[0]

        print(
            "\nUsing explicitly provided "
            "training dataset:"
        )

        print(
            f"  • {train_file.name}"
        )

        train_data = self._read_dataset(
            train_file
        )

        # --------------------------------------------------------
        # Validation dataset
        # --------------------------------------------------------

        validation_data = None

        if validation_files:

            if len(validation_files) > 1:

                raise ValueError(
                    "\nMultiple validation datasets detected:\n"
                    + "\n".join(
                        f"  • {file.name}"
                        for file in validation_files
                    )
                    + "\n\n"
                    "Please provide a single "
                    "validation dataset."
                )

            validation_file = (
                validation_files[0]
            )

            print(
                "\nUsing explicitly provided "
                "validation dataset:"
            )

            print(
                f"  • {validation_file.name}"
            )

            validation_data = (
                self._read_dataset(
                    validation_file
                )
            )

        # --------------------------------------------------------
        # Test dataset
        # --------------------------------------------------------

        if not test_files:

            raise ValueError(
                "\nA training dataset was explicitly "
                "provided, but no test dataset was found.\n\n"
                "Please provide a file such as:\n"
                "  test.csv\n"
                "  house_test.csv"
            )

        if len(test_files) > 1:

            raise ValueError(
                "\nMultiple testing datasets detected:\n"
                + "\n".join(
                    f"  • {file.name}"
                    for file in test_files
                )
                + "\n\n"
                "Please provide a single "
                "testing dataset."
            )

        test_file = test_files[0]

        print(
            "\nUsing explicitly provided "
            "testing dataset:"
        )

        print(
            f"  • {test_file.name}"
        )

        test_data = self._read_dataset(
            test_file
        )

        return (
            train_data,
            validation_data,
            test_data,
        )

    # ============================================================
    # Target Detection
    # ============================================================

    @staticmethod
    def _detect_target_column(
        data: pd.DataFrame,
    ) -> str:
        """
        Detect the target column.

        Current MVP rule:
            Last column = target.

        This will later be replaced with
        intelligent target detection.
        """

        if data.empty:

            raise ValueError(
                "Cannot detect target from "
                "an empty dataset."
            )

        if len(data.columns) < 2:

            raise ValueError(
                "Dataset must contain at least "
                "two columns."
            )

        target_column = data.columns[-1]

        return str(
            target_column
        )

    # ============================================================
    # Validate Dataset Compatibility
    # ============================================================

    @staticmethod
    def _check_column_consistency(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        validation_data: Optional[
            pd.DataFrame
        ] = None,
    ) -> None:
        """
        Ensure train/test/validation datasets
        contain compatible columns.

        This is a basic consistency check.
        Full validation will be handled later
        by data_validation.py.
        """

        train_columns = list(
            train_data.columns
        )

        test_columns = list(
            test_data.columns
        )

        if train_columns != test_columns:

            raise ValueError(
                "\nTraining and testing datasets "
                "have different columns.\n\n"
                f"Training columns:\n"
                f"{train_columns}\n\n"
                f"Testing columns:\n"
                f"{test_columns}"
            )

        if validation_data is not None:

            validation_columns = list(
                validation_data.columns
            )

            if (
                train_columns
                != validation_columns
            ):

                raise ValueError(
                    "\nTraining and validation "
                    "datasets have different columns.\n\n"
                    f"Training columns:\n"
                    f"{train_columns}\n\n"
                    f"Validation columns:\n"
                    f"{validation_columns}"
                )

    # ============================================================
    # Save Dataset
    # ============================================================

    def _save_dataset(
        self,
        data: pd.DataFrame,
        filename: str,
    ) -> Path:
        """
        Save a dataset into artifacts/.
        """

        self.artifacts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = (
            self.artifacts_dir
            / filename
        )

        data.to_csv(
            path,
            index=False,
        )

        return path

    # ============================================================
    # Main Ingestion
    # ============================================================

    def initiate_data_ingestion(
        self,
    ) -> Tuple[
        pd.DataFrame,
        Optional[pd.DataFrame],
        pd.DataFrame,
    ]:
        """
        Perform complete data ingestion.

        Returns
        -------
        tuple
            train_data,
            validation_data,
            test_data
        """

        print(
            "\n[1/6] Data Ingestion"
        )

        # --------------------------------------------------------
        # Analyze available files
        # --------------------------------------------------------

        analyzer = DataAnalyzer(
            self.project_root
        )

        analysis = analyzer.analyze()

        train_files = analysis[
            "train_files"
        ]

        validation_files = analysis[
            "validation_files"
        ]

        test_files = analysis[
            "test_files"
        ]

        # --------------------------------------------------------
        # Determine ingestion strategy
        # --------------------------------------------------------
        #
        # If explicit train/test files exist,
        # DO NOT perform random splitting.
        #
        # Otherwise use the normal dataset workflow.
        # --------------------------------------------------------

        has_explicit_split = bool(
            train_files
            or validation_files
            or test_files
        )

        # --------------------------------------------------------
        # Explicit train/test/validation
        # --------------------------------------------------------

        if has_explicit_split:

            # If only one special file exists,
            # we should not silently guess.
            #
            # Example:
            #
            # train.csv
            #
            # but no test.csv.
            #
            # This must be handled explicitly.

            if not train_files:

                raise ValueError(
                    "\nAn explicit dataset split was detected "
                    "but no training dataset was found."
                )

            if not test_files:

                raise ValueError(
                    "\nAn explicit training dataset was detected "
                    "but no testing dataset was found.\n\n"
                    "Please add a file such as:\n"
                    "  test.csv\n"
                    "  house_test.csv"
                )

            (
                train_data,
                validation_data,
                test_data,
            ) = (
                self._prepare_explicit_datasets(
                    analysis
                )
            )

        # --------------------------------------------------------
        # Normal dataset workflow
        # --------------------------------------------------------

        else:

            data = (
                self._prepare_normal_dataset(
                    analysis
                )
            )

            # ----------------------------------------------------
            # Target detection
            # ----------------------------------------------------

            target_column = (
                self._detect_target_column(
                    data
                )
            )

            print(
                f"\nTarget column detected: "
                f"{target_column}"
            )

            # ----------------------------------------------------
            # Automatic train/test split
            # ----------------------------------------------------

            train_data, test_data = (
                train_test_split(
                    data,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
            )

            # ----------------------------------------------------
            # No validation dataset in the
            # basic automatic workflow yet.
            # ----------------------------------------------------

            validation_data = None

        # --------------------------------------------------------
        # Basic consistency check
        # --------------------------------------------------------

        self._check_column_consistency(
            train_data,
            test_data,
            validation_data,
        )

        # --------------------------------------------------------
        # Target detection from training data
        # --------------------------------------------------------

        target_column = (
            self._detect_target_column(
                train_data
            )
        )

        print(
            f"\nTarget column detected: "
            f"{target_column}"
        )

        # --------------------------------------------------------
        # Save train
        # --------------------------------------------------------

        train_path = self._save_dataset(
            train_data,
            "train.csv",
        )

        # --------------------------------------------------------
        # Save validation
        # --------------------------------------------------------

        validation_path = None

        if validation_data is not None:

            validation_path = (
                self._save_dataset(
                    validation_data,
                    "validation.csv",
                )
            )

        # --------------------------------------------------------
        # Save test
        # --------------------------------------------------------

        test_path = self._save_dataset(
            test_data,
            "test.csv",
        )

        # --------------------------------------------------------
        # Display summary
        # --------------------------------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "DATA INGESTION SUMMARY"
        )

        print(
            "=" * 60
        )

        print(
            f"\nTraining samples   : "
            f"{len(train_data)}"
        )

        if validation_data is not None:

            print(
                f"Validation samples : "
                f"{len(validation_data)}"
            )

        else:

            print(
                "Validation samples : "
                "Not provided"
            )

        print(
            f"Testing samples    : "
            f"{len(test_data)}"
        )

        print(
            f"\nTraining data saved:"
            f"\n{train_path}"
        )

        if validation_path:

            print(
                f"\nValidation data saved:"
                f"\n{validation_path}"
            )

        print(
            f"\nTesting data saved:"
            f"\n{test_path}"
        )

        print(
            "\nData ingestion completed successfully."
        )

        print(
            "=" * 60
        )

        return (
            train_data,
            validation_data,
            test_data,
        )
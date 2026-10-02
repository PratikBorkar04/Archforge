"""
data_ingestion.py

NLP Data Ingestion module for ArchForge.

Responsibilities
----------------
1. Discover CSV datasets from data/raw/.
2. Allow a maximum of 10 CSV files.
3. Detect duplicate files.
4. Recognize explicit train/test/validation files.
5. Preserve explicitly provided splits.
6. Combine compatible normal datasets.
7. Handle multiple unrelated datasets safely.
8. Automatically create an 80/20 train/test split when needed.
9. Check train/test/validation schema consistency.
10. Save processed datasets into artifacts/.

Important
---------
- File order is NEVER used to determine train/test/validation roles.
- Target detection is NOT performed here.
- Text-column detection is NOT performed here.
- Those decisions belong to later NLP stages.
"""

from pathlib import Path
from typing import Optional, Tuple
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split


class NLPDataIngestion:
    """
    Discover and prepare datasets for the NLP pipeline.
    """

    MAX_CSV_FILES = 10

    TRAIN_KEYWORDS = (
        "train",
        "training",
    )

    VALIDATION_KEYWORDS = (
        "validation",
        "valid",
        "val",
    )

    TEST_KEYWORDS = (
        "test",
        "testing",
    )

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
            self.project_root
            / "data"
            / "raw"
        )

        self.artifacts_dir = (
            self.project_root
            / "artifacts"
        )

        self.test_size = test_size
        self.random_state = random_state

    # ============================================================
    # Discover CSV Files
    # ============================================================

    def _discover_csv_files(self):
        """
        Discover CSV files from data/raw/.

        Maximum allowed:
            10 CSV files.
        """

        if not self.raw_data_dir.exists():

            raise FileNotFoundError(
                f"\nRaw data directory does not exist:\n"
                f"{self.raw_data_dir}"
            )

        csv_files = sorted(
            self.raw_data_dir.glob("*.csv")
        )

        if not csv_files:

            raise FileNotFoundError(
                f"\nNo CSV files found in:\n"
                f"{self.raw_data_dir}"
            )

        print(
            f"\nFound {len(csv_files)} CSV file(s)."
        )

        if len(csv_files) > self.MAX_CSV_FILES:

            raise ValueError(
                "\nToo many CSV files detected.\n"
                f"Maximum allowed: {self.MAX_CSV_FILES}\n"
                f"Found: {len(csv_files)}\n\n"
                "Please keep a maximum of 10 CSV files "
                "inside data/raw/."
            )

        for file in csv_files:

            print(
                f"  • {file.name}"
            )

        return csv_files

    # ============================================================
    # Duplicate File Detection
    # ============================================================

    @staticmethod
    def _file_hash(
        file: Path,
    ) -> str:
        """
        Generate SHA-256 hash for a file.

        This detects files containing exactly
        the same content even when their names differ.
        """

        sha256 = hashlib.sha256()

        with file.open(
            "rb"
        ) as handle:

            for chunk in iter(
                lambda: handle.read(1024 * 1024),
                b"",
            ):

                sha256.update(chunk)

        return sha256.hexdigest()

    def _check_duplicate_files(
        self,
        files,
    ) -> None:
        """
        Detect duplicate CSV files using file content.
        """

        hashes = {}

        duplicates = []

        for file in files:

            file_hash = self._file_hash(
                file
            )

            if file_hash in hashes:

                duplicates.append(
                    (
                        hashes[file_hash],
                        file,
                    )
                )

            else:

                hashes[file_hash] = file

        if duplicates:

            print(
                "\n" + "=" * 60
            )

            print(
                "DUPLICATE DATASETS DETECTED"
            )

            print(
                "=" * 60
            )

            for original, duplicate in duplicates:

                print(
                    f"\nDuplicate files:"
                    f"\n  • {original.name}"
                    f"\n  • {duplicate.name}"
                )

            raise ValueError(
                "\nDuplicate CSV files detected.\n"
                "Please remove duplicate files before "
                "starting NLP ingestion."
            )

        print(
            "\nNo duplicate CSV files detected."
        )

    # ============================================================
    # Read Dataset
    # ============================================================

    @staticmethod
    def _read_dataset(
        file: Path,
    ) -> pd.DataFrame:
        """
        Read a CSV dataset safely.
        """

        try:

            data = pd.read_csv(
                file
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

        print(
            f"\nLoaded: {file.name}"
            f" | Rows: {len(data)}"
            f" | Columns: {len(data.columns)}"
        )

        return data

    # ============================================================
    # Detect Explicit Dataset Role
    # ============================================================

    @classmethod
    def _detect_file_role(
        cls,
        file: Path,
    ) -> Optional[str]:
        """
        Detect whether a filename represents:

            train
            validation
            test

        Examples:
            train.csv
            training.csv
            house_train.csv
            train_data.csv
            review_validation.csv
            review_val.csv
            review_test.csv

        Returns:
            'train'
            'validation'
            'test'
            None
        """

        name = file.stem.lower()

        # Convert common separators into spaces.
        normalized = (
            name
            .replace("-", " ")
            .replace("_", " ")
            .replace(".", " ")
        )

        tokens = set(
            normalized.split()
        )

        if tokens.intersection(
            cls.TRAIN_KEYWORDS
        ):

            return "train"

        if tokens.intersection(
            cls.VALIDATION_KEYWORDS
        ):

            return "validation"

        if tokens.intersection(
            cls.TEST_KEYWORDS
        ):

            return "test"

        return None

    # ============================================================
    # Classify Files
    # ============================================================

    @classmethod
    def _classify_files(
        cls,
        files,
    ):
        """
        Separate explicit split files from
        normal datasets.
        """

        train_files = []
        validation_files = []
        test_files = []
        normal_files = []

        for file in files:

            role = cls._detect_file_role(
                file
            )

            if role == "train":

                train_files.append(file)

            elif role == "validation":

                validation_files.append(file)

            elif role == "test":

                test_files.append(file)

            else:

                normal_files.append(file)

        return (
            train_files,
            validation_files,
            test_files,
            normal_files,
        )

    # ============================================================
    # Validate Explicit Split Configuration
    # ============================================================

    @staticmethod
    def _validate_explicit_split(
        train_files,
        validation_files,
        test_files,
        normal_files,
    ) -> None:
        """
        Validate explicitly supplied train/test/validation files.
        """

        # --------------------------------------------------------
        # Multiple files for the same explicit role
        # --------------------------------------------------------

        if len(train_files) > 1:

            raise ValueError(
                "\nMultiple training files detected:\n"
                + "\n".join(
                    f"  • {file.name}"
                    for file in train_files
                )
                + "\n\n"
                "Please provide only one training file."
            )

        if len(validation_files) > 1:

            raise ValueError(
                "\nMultiple validation files detected:\n"
                + "\n".join(
                    f"  • {file.name}"
                    for file in validation_files
                )
                + "\n\n"
                "Please provide only one validation file."
            )

        if len(test_files) > 1:

            raise ValueError(
                "\nMultiple testing files detected:\n"
                + "\n".join(
                    f"  • {file.name}"
                    for file in test_files
                )
                + "\n\n"
                "Please provide only one testing file."
            )

        # --------------------------------------------------------
        # Explicit split must contain train + test
        # --------------------------------------------------------

        if train_files and not test_files:

            raise ValueError(
                "\nAn explicit training dataset was detected, "
                "but no testing dataset was found.\n\n"
                "Expected something like:\n"
                "  train.csv\n"
                "  test.csv"
            )

        if test_files and not train_files:

            raise ValueError(
                "\nAn explicit testing dataset was detected, "
                "but no training dataset was found."
            )

        # --------------------------------------------------------
        # Validation alone is not meaningful
        # --------------------------------------------------------

        if validation_files and not (
            train_files and test_files
        ):

            raise ValueError(
                "\nA validation dataset was detected, "
                "but train/test datasets are incomplete."
            )

        # --------------------------------------------------------
        # Do not silently mix explicit splits with
        # unrelated normal datasets.
        # --------------------------------------------------------

        if (
            train_files
            or validation_files
            or test_files
        ) and normal_files:

            raise ValueError(
                "\nMixed dataset configuration detected.\n\n"
                "Explicit train/test/validation files were found "
                "together with normal CSV files:\n\n"
                + "\n".join(
                    f"  • {file.name}"
                    for file in normal_files
                )
                + "\n\n"
                "Please keep either:\n"
                "  1. Explicit train/validation/test files\n"
                "or\n"
                "  2. Normal dataset files\n\n"
                "ArchForge will not guess which files should be used."
            )

    # ============================================================
    # Combine Same-Schema Files
    # ============================================================

    def _combine_same_schema_files(
        self,
        files,
    ) -> pd.DataFrame:
        """
        Combine multiple normal CSV files
        having exactly the same columns.
        """

        datasets = []

        reference_columns = None

        print(
            "\nCombining compatible datasets..."
        )

        for file in files:

            data = self._read_dataset(
                file
            )

            columns = list(
                data.columns
            )

            if reference_columns is None:

                reference_columns = columns

            elif columns != reference_columns:

                raise ValueError(
                    "\nSchema mismatch while combining datasets.\n\n"
                    f"Expected columns:\n"
                    f"{reference_columns}\n\n"
                    f"File: {file.name}\n"
                    f"Columns:\n"
                    f"{columns}"
                )

            datasets.append(
                data
            )

        combined = pd.concat(
            datasets,
            ignore_index=True,
        )

        print(
            f"\nCombined dataset:"
            f" {len(combined)} rows × "
            f"{len(combined.columns)} columns"
        )

        return combined

    # ============================================================
    # Find Dataset Groups
    # ============================================================

    def _group_normal_datasets(
        self,
        files,
    ):
        """
        Group normal datasets by exact column schema.
        """

        groups = []

        for file in files:

            data = self._read_dataset(
                file
            )

            columns = tuple(
                data.columns
            )

            matched_group = None

            for group in groups:

                if group["columns"] == columns:

                    matched_group = group
                    break

            if matched_group is None:

                groups.append(
                    {
                        "columns": columns,
                        "files": [file],
                    }
                )

            else:

                matched_group[
                    "files"
                ].append(file)

        return [
            group["files"]
            for group in groups
        ]

    # ============================================================
    # Ask User Which Dataset Group To Use
    # ============================================================

    @staticmethod
    def _ask_dataset_choice(
        dataset_groups,
    ):
        """
        Ask the user which independent dataset group
        should be used.
        """

        print(
            "\n" + "=" * 60
        )

        print(
            "MULTIPLE DATASET GROUPS DETECTED"
        )

        print(
            "=" * 60
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
                "\nSelect dataset group: "
            ).strip()

            if choice == "0":

                raise RuntimeError(
                    "NLP data ingestion cancelled by user."
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
                "Invalid selection."
            )

    # ============================================================
    # Check Column Consistency
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
        contain the same columns in the same order.
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
                f"Training:\n{train_columns}\n\n"
                f"Testing:\n{test_columns}"
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
                    "\nTraining and validation datasets "
                    "have different columns.\n\n"
                    f"Training:\n{train_columns}\n\n"
                    f"Validation:\n{validation_columns}"
                )

        print(
            "\nSchema consistency check passed."
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
        Save dataset to artifacts/.
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

        print(
            f"Saved: {path}"
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
        Perform NLP dataset ingestion.

        Returns
        -------
        tuple
            train_data,
            validation_data,
            test_data
        """

        print(
            "\n" + "=" * 60
        )

        print(
            "NLP DATA INGESTION"
        )

        print(
            "=" * 60
        )

        # --------------------------------------------------------
        # 1. Discover files
        # --------------------------------------------------------

        files = self._discover_csv_files()

        # --------------------------------------------------------
        # 2. Duplicate detection
        # --------------------------------------------------------

        self._check_duplicate_files(
            files
        )

        # --------------------------------------------------------
        # 3. Classify files
        # --------------------------------------------------------

        (
            train_files,
            validation_files,
            test_files,
            normal_files,
        ) = self._classify_files(
            files
        )

        # --------------------------------------------------------
        # Print detected roles
        # --------------------------------------------------------

        print(
            "\nDetected dataset roles:"
        )

        print(
            f"  Training   : "
            f"{[file.name for file in train_files]}"
        )

        print(
            f"  Validation : "
            f"{[file.name for file in validation_files]}"
        )

        print(
            f"  Testing    : "
            f"{[file.name for file in test_files]}"
        )

        print(
            f"  Normal     : "
            f"{[file.name for file in normal_files]}"
        )

        # --------------------------------------------------------
        # 4. Validate split configuration
        # --------------------------------------------------------

        self._validate_explicit_split(
            train_files,
            validation_files,
            test_files,
            normal_files,
        )

        # ========================================================
        # EXPLICIT TRAIN / VALIDATION / TEST
        # ========================================================

        if train_files:

            print(
                "\nUsing explicitly provided "
                "train/validation/test datasets."
            )

            train_data = self._read_dataset(
                train_files[0]
            )

            test_data = self._read_dataset(
                test_files[0]
            )

            validation_data = None

            if validation_files:

                validation_data = (
                    self._read_dataset(
                        validation_files[0]
                    )
                )

        # ========================================================
        # NORMAL DATASET WORKFLOW
        # ========================================================

        else:

            if not normal_files:

                raise ValueError(
                    "\nNo usable NLP dataset found."
                )

            # ----------------------------------------------------
            # One normal dataset
            # ----------------------------------------------------

            if len(normal_files) == 1:

                print(
                    f"\nUsing dataset:"
                    f" {normal_files[0].name}"
                )

                data = self._read_dataset(
                    normal_files[0]
                )

            # ----------------------------------------------------
            # Multiple normal datasets
            # ----------------------------------------------------

            else:

                groups = (
                    self._group_normal_datasets(
                        normal_files
                    )
                )

                if len(groups) == 1:

                    data = (
                        self._combine_same_schema_files(
                            groups[0]
                        )
                    )

                else:

                    selected_group = (
                        self._ask_dataset_choice(
                            groups
                        )
                    )

                    data = (
                        self._combine_same_schema_files(
                            selected_group
                        )
                    )

            # ----------------------------------------------------
            # Automatic train/test split
            # ----------------------------------------------------

            print(
                "\nCreating automatic 80/20 "
                "train/test split..."
            )

            train_data, test_data = (
                train_test_split(
                    data,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
            )

            validation_data = None

        # --------------------------------------------------------
        # 5. Schema consistency
        # --------------------------------------------------------

        self._check_column_consistency(
            train_data,
            test_data,
            validation_data,
        )

        # --------------------------------------------------------
        # 6. Save artifacts
        # --------------------------------------------------------

        train_path = self._save_dataset(
            train_data,
            "train.csv",
        )

        validation_path = None

        if validation_data is not None:

            validation_path = (
                self._save_dataset(
                    validation_data,
                    "validation.csv",
                )
            )

        test_path = self._save_dataset(
            test_data,
            "test.csv",
        )

        # --------------------------------------------------------
        # 7. Summary
        # --------------------------------------------------------

        print(
            "\n" + "=" * 60
        )

        print(
            "NLP DATA INGESTION SUMMARY"
        )

        print(
            "=" * 60
        )

        print(
            f"\nTraining samples   : "
            f"{len(train_data)}"
        )

        print(
            f"Validation samples : "
            f"{len(validation_data) if validation_data is not None else 'Not provided'}"
        )

        print(
            f"Testing samples    : "
            f"{len(test_data)}"
        )

        print(
            f"\nTraining file:"
            f"\n{train_path}"
        )

        if validation_path:

            print(
                f"\nValidation file:"
                f"\n{validation_path}"
            )

        print(
            f"\nTesting file:"
            f"\n{test_path}"
        )

        print(
            "\nNLP data ingestion completed successfully."
        )

        print(
            "=" * 60
        )

        return (
            train_data,
            validation_data,
            test_data,
        )
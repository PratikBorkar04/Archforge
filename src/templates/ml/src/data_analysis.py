"""
data_analysis.py

Dataset discovery and analysis module for ArchForge.

Responsibilities
----------------
1. Discover CSV files from data/raw/.
2. Enforce a maximum file limit.
3. Detect explicit train/test/validation files.
4. Detect files that belong to the same dataset.
5. Detect potentially related datasets.
6. Identify possible join keys.
7. Prevent accidental merging of unrelated datasets.

Important
---------
File order is NEVER used to determine whether a file is
training or testing data.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ================================================================
# Configuration
# ================================================================

MAX_DATA_FILES = 10

SUPPORTED_FORMATS = {
    ".csv",
}


# ================================================================
# Dataset role names
# ================================================================

TRAIN_NAMES = {
    "train",
    "training",
}

TEST_NAMES = {
    "test",
    "testing",
}

VALIDATION_NAMES = {
    "val",
    "valid",
    "validation",
}


# ================================================================
# Files that should normally be ignored
# ================================================================

IGNORED_NAMES = {
    "metadata",
    "meta",
    "readme",
    "documentation",
    "docs",
    "info",
}


class DataAnalyzer:
    """
    Analyze dataset files available in data/raw/.
    """

    def __init__(self, project_root: Path) -> None:

        self.project_root = Path(
            project_root
        ).resolve()

        self.raw_data_dir = (
            self.project_root
            / "data"
            / "raw"
        )

    # ============================================================
    # File Discovery
    # ============================================================

    def discover_files(self) -> List[Path]:
        """
        Discover supported dataset files.

        Returns
        -------
        List[Path]
            Discovered CSV files.
        """

        if not self.raw_data_dir.exists():

            raise FileNotFoundError(
                "\nRaw data directory was not found:\n"
                f"{self.raw_data_dir}\n\n"
                "Please create the directory and "
                "place your dataset inside it."
            )

        files = [
            file
            for file in self.raw_data_dir.iterdir()
            if (
                file.is_file()
                and file.suffix.lower()
                in SUPPORTED_FORMATS
            )
        ]

        if not files:

            raise FileNotFoundError(
                "\nNo CSV dataset found.\n\n"
                "Please place your CSV file inside:\n"
                f"{self.raw_data_dir}"
            )

        # --------------------------------------------------------
        # File limit
        # --------------------------------------------------------

        if len(files) > MAX_DATA_FILES:

            raise ValueError(
                "\nToo many dataset files detected.\n\n"
                f"ArchForge currently supports a maximum "
                f"of {MAX_DATA_FILES} CSV files.\n\n"
                f"Files detected: {len(files)}"
            )

        return sorted(files)

    # ============================================================
    # Filename Normalization
    # ============================================================

    @staticmethod
    def normalize_filename(
        file: Path,
    ) -> str:
        """
        Normalize a filename for analysis.

        Example
        -------
        House Price-Train.csv

        becomes approximately:

        house_price_train
        """

        return (
            file.stem
            .lower()
            .strip()
            .replace("-", "_")
            .replace(" ", "_")
        )

    # ============================================================
    # Filename Parts
    # ============================================================

    @staticmethod
    def get_filename_parts(
        file: Path,
    ) -> List[str]:
        """
        Split normalized filename into meaningful parts.
        """

        name = DataAnalyzer.normalize_filename(
            file
        )

        return [
            part
            for part in name.split("_")
            if part
        ]

    # ============================================================
    # Dataset Role Detection
    # ============================================================

    @staticmethod
    def detect_dataset_role(
        file: Path,
    ) -> str:
        """
        Detect whether a file is explicitly marked as:

            train
            test
            validation
            other
            ignored

        Examples
        --------
        train.csv
        house_train.csv
        customer_training_data.csv

        are recognized as training files.

        The position/order of files has NO effect.
        """

        parts = set(
            DataAnalyzer.get_filename_parts(
                file
            )
        )

        # --------------------------------------------------------
        # Ignore metadata/documentation files
        # --------------------------------------------------------

        if parts.intersection(
            IGNORED_NAMES
        ):

            return "ignored"

        # --------------------------------------------------------
        # Training
        # --------------------------------------------------------

        if parts.intersection(
            TRAIN_NAMES
        ):

            return "train"

        # --------------------------------------------------------
        # Testing
        # --------------------------------------------------------

        if parts.intersection(
            TEST_NAMES
        ):

            return "test"

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        if parts.intersection(
            VALIDATION_NAMES
        ):

            return "validation"

        # --------------------------------------------------------
        # Normal dataset
        # --------------------------------------------------------

        return "other"

    # ============================================================
    # Classify Files
    # ============================================================

    def classify_files(
        self,
        files: List[Path],
    ) -> Dict[str, List[Path]]:
        """
        Classify all discovered files.

        Returns
        -------
        dict
            Files grouped by role.
        """

        result = {
            "train": [],
            "test": [],
            "validation": [],
            "other": [],
            "ignored": [],
        }

        for file in files:

            role = self.detect_dataset_role(
                file
            )

            result[role].append(file)

        return result

    # ============================================================
    # Read Dataset
    # ============================================================

    @staticmethod
    def read_dataset(
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
                f"{file.name}\n\n"
                f"Reason: {exc}"
            ) from exc

        if data.empty:

            raise ValueError(
                f"\nDataset is empty:\n"
                f"{file.name}"
            )

        return data

    # ============================================================
    # Normalize Column Names
    # ============================================================

    @staticmethod
    def normalize_column_name(
        column: str,
    ) -> str:
        """
        Normalize one column name for comparison.

        Original column names are NOT modified.
        """

        return (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

    # ============================================================
    # Get Normalized Columns
    # ============================================================

    @classmethod
    def get_normalized_columns(
        cls,
        data: pd.DataFrame,
    ) -> List[str]:
        """
        Return normalized column names.
        """

        return sorted(
            cls.normalize_column_name(
                column
            )
            for column in data.columns
        )

    # ============================================================
    # Compare Schemas
    # ============================================================

    @classmethod
    def same_schema(
        cls,
        first: pd.DataFrame,
        second: pd.DataFrame,
    ) -> bool:
        """
        Determine whether two datasets have the same schema.

        Checks:

        1. Same number of columns.
        2. Same normalized column names.
        3. Compatible data types.

        If all conditions match, the datasets can
        potentially be concatenated.
        """

        first_columns = (
            cls.get_normalized_columns(
                first
            )
        )

        second_columns = (
            cls.get_normalized_columns(
                second
            )
        )

        if first_columns != second_columns:

            return False

        # --------------------------------------------------------
        # Data type comparison
        # --------------------------------------------------------

        first_types = {
            cls.normalize_column_name(
                column
            ): str(dtype)
            for column, dtype
            in first.dtypes.items()
        }

        second_types = {
            cls.normalize_column_name(
                column
            ): str(dtype)
            for column, dtype
            in second.dtypes.items()
        }

        return first_types == second_types

    # ============================================================
    # Common Columns
    # ============================================================

    @classmethod
    def find_common_columns(
        cls,
        first: pd.DataFrame,
        second: pd.DataFrame,
    ) -> List[str]:
        """
        Find columns that appear in both datasets.
        """

        first_columns = {
            cls.normalize_column_name(
                column
            ): column
            for column in first.columns
        }

        second_columns = {
            cls.normalize_column_name(
                column
            ): column
            for column in second.columns
        }

        common_columns = []

        for normalized_name in first_columns:

            if normalized_name in second_columns:

                common_columns.append(
                    first_columns[
                        normalized_name
                    ]
                )

        return common_columns

    # ============================================================
    # Detect Join Key
    # ============================================================

    @classmethod
    def find_join_key(
        cls,
        first: pd.DataFrame,
        second: pd.DataFrame,
    ) -> Optional[str]:
        """
        Detect a possible meaningful join key.

        Strong candidates include:

            id
            house_id
            customer_id
            product_id
            order_id

        Returns
        -------
        str or None
        """

        common_columns = (
            cls.find_common_columns(
                first,
                second,
            )
        )

        for column in common_columns:

            normalized = (
                cls.normalize_column_name(
                    column
                )
            )

            # ----------------------------------------------------
            # Strong ID candidates
            # ----------------------------------------------------

            if normalized == "id":

                return column

            if normalized.endswith(
                "_id"
            ):

                return column

            if normalized.endswith(
                "id"
            ):

                return column

        return None

    # ============================================================
    # Relationship Analysis
    # ============================================================

    def analyze_relationship(
        self,
        first_file: Path,
        second_file: Path,
    ) -> str:
        """
        Determine the relationship between two datasets.

        Returns
        -------
        str

            same_schema
                Same dataset structure.
                Usually concatenate.

            related
                Different schemas but possible
                common join key.

            unrelated
                No meaningful relationship detected.
        """

        first = self.read_dataset(
            first_file
        )

        second = self.read_dataset(
            second_file
        )

        # --------------------------------------------------------
        # Same schema
        # --------------------------------------------------------

        if self.same_schema(
            first,
            second,
        ):

            return "same_schema"

        # --------------------------------------------------------
        # Possible relationship
        # --------------------------------------------------------

        join_key = self.find_join_key(
            first,
            second,
        )

        if join_key is not None:

            return "related"

        # --------------------------------------------------------
        # Unrelated
        # --------------------------------------------------------

        return "unrelated"

    # ============================================================
    # Group Same-Schema Files
    # ============================================================

    def group_same_schema_files(
        self,
        files: List[Path],
    ) -> List[List[Path]]:
        """
        Group files having the same schema.

        Example
        -------

        house1.csv
        house2.csv
        house3.csv

        with identical schemas become:

        [
            [
                house1.csv,
                house2.csv,
                house3.csv
            ]
        ]
        """

        groups: List[List[Path]] = []

        used_files = set()

        for file in files:

            if file in used_files:

                continue

            current_group = [
                file
            ]

            used_files.add(file)

            first_data = self.read_dataset(
                file
            )

            for other_file in files:

                if other_file in used_files:

                    continue

                other_data = (
                    self.read_dataset(
                        other_file
                    )
                )

                if self.same_schema(
                    first_data,
                    other_data,
                ):

                    current_group.append(
                        other_file
                    )

                    used_files.add(
                        other_file
                    )

            groups.append(
                current_group
            )

        return groups

    # ============================================================
    # Analyze All Relationships
    # ============================================================

    def analyze_relationships(
        self,
        files: List[Path],
    ) -> List[Dict]:
        """
        Analyze pairwise relationships between datasets.
        """

        relationships = []

        for index in range(
            len(files)
        ):

            for next_index in range(
                index + 1,
                len(files),
            ):

                first = files[index]

                second = files[
                    next_index
                ]

                relationship = (
                    self.analyze_relationship(
                        first,
                        second,
                    )
                )

                join_key = None

                if relationship == "related":

                    first_data = (
                        self.read_dataset(
                            first
                        )
                    )

                    second_data = (
                        self.read_dataset(
                            second
                        )
                    )

                    join_key = (
                        self.find_join_key(
                            first_data,
                            second_data,
                        )
                    )

                relationships.append(
                    {
                        "first": first,
                        "second": second,
                        "relationship": relationship,
                        "join_key": join_key,
                    }
                )

        return relationships

    # ============================================================
    # Print Analysis
    # ============================================================

    def print_analysis(
        self,
        analysis: Dict,
    ) -> None:
        """
        Display dataset analysis in the terminal.
        """

        print(
            "\n"
            + "=" * 60
        )

        print(
            "DATASET ANALYSIS"
        )

        print(
            "=" * 60
        )

        all_files = analysis[
            "all_files"
        ]

        print(
            f"\nCSV files detected: "
            f"{len(all_files)}"
        )

        for file in all_files:

            print(
                f"  • {file.name}"
            )

        # --------------------------------------------------------
        # Explicit train
        # --------------------------------------------------------

        if analysis[
            "train_files"
        ]:

            print(
                "\nExplicit training files:"
            )

            for file in analysis[
                "train_files"
            ]:

                print(
                    f"  • {file.name}"
                )

        # --------------------------------------------------------
        # Explicit validation
        # --------------------------------------------------------

        if analysis[
            "validation_files"
        ]:

            print(
                "\nExplicit validation files:"
            )

            for file in analysis[
                "validation_files"
            ]:

                print(
                    f"  • {file.name}"
                )

        # --------------------------------------------------------
        # Explicit test
        # --------------------------------------------------------

        if analysis[
            "test_files"
        ]:

            print(
                "\nExplicit testing files:"
            )

            for file in analysis[
                "test_files"
            ]:

                print(
                    f"  • {file.name}"
                )

        # --------------------------------------------------------
        # Ignored
        # --------------------------------------------------------

        if analysis[
            "ignored_files"
        ]:

            print(
                "\nIgnored files:"
            )

            for file in analysis[
                "ignored_files"
            ]:

                print(
                    f"  • {file.name}"
                )

        # --------------------------------------------------------
        # Dataset groups
        # --------------------------------------------------------

        groups = analysis[
            "dataset_groups"
        ]

        if groups:

            print(
                "\nDataset groups:"
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

        # --------------------------------------------------------
        # Relationships
        # --------------------------------------------------------

        relationships = analysis[
            "relationships"
        ]

        if relationships:

            print(
                "\nDataset relationships:"
            )

            for relationship in relationships:

                first = relationship[
                    "first"
                ].name

                second = relationship[
                    "second"
                ].name

                relation = relationship[
                    "relationship"
                ]

                print(
                    f"\n{first}"
                )

                print(
                    f"  ↕ {relation}"
                )

                print(
                    f"{second}"
                )

                join_key = relationship[
                    "join_key"
                ]

                if join_key:

                    print(
                        f"  Possible join key: "
                        f"{join_key}"
                    )

        print(
            "\n"
            + "=" * 60
        )

    # ============================================================
    # Complete Analysis
    # ============================================================

    def analyze(self) -> Dict:
        """
        Perform complete dataset analysis.

        Returns
        -------
        dict
            Complete analysis result.
        """

        # --------------------------------------------------------
        # Discover
        # --------------------------------------------------------

        files = self.discover_files()

        # --------------------------------------------------------
        # Classify
        # --------------------------------------------------------

        classified = self.classify_files(
            files
        )

        # --------------------------------------------------------
        # Only normal datasets are candidates
        # for automatic grouping.
        # --------------------------------------------------------

        other_files = classified[
            "other"
        ]

        # --------------------------------------------------------
        # Group same-schema datasets
        # --------------------------------------------------------

        dataset_groups = (
            self.group_same_schema_files(
                other_files
            )
        )

        # --------------------------------------------------------
        # Analyze relationships
        # --------------------------------------------------------

        relationships = (
            self.analyze_relationships(
                other_files
            )
        )

        # --------------------------------------------------------
        # Build result
        # --------------------------------------------------------

        analysis = {
            "all_files": files,

            "train_files": classified[
                "train"
            ],

            "test_files": classified[
                "test"
            ],

            "validation_files": classified[
                "validation"
            ],

            "other_files": other_files,

            "ignored_files": classified[
                "ignored"
            ],

            "dataset_groups": dataset_groups,

            "relationships": relationships,
        }

        # --------------------------------------------------------
        # Display
        # --------------------------------------------------------

        self.print_analysis(
            analysis
        )

        return analysis
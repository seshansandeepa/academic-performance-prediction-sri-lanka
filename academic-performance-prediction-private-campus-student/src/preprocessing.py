"""
Final preprocessing workflow for the academic performance
prediction research project.

The raw participant dataset is private and is NOT included
in the public GitHub repository.

Supported input formats:
- Excel (.xlsx)
- CSV (.csv)
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

RAW_FOLDER = Path("data/raw")
PROCESSED_FOLDER = Path("data/processed")

OUTPUT_FILE = PROCESSED_FOLDER / "cleaned_survey_data.csv"
REPORT_FILE = PROCESSED_FOLDER / "preprocessing_report.txt"

TARGET = "academic_performance_level"


# ---------------------------------------------------------
# SURVEY QUESTION NAMES
# ---------------------------------------------------------

QUESTION_NAMES = {
    1: "consent",
    2: "currently_studying",
    3: "institution_type",
    4: "year_of_study",
    5: "degree_area",
    6: "study_mode",
    7: "gender",
    8: "attendance",
    9: "study_hours",
    10: "lms_usage",
    11: "assignment_habit",
    12: "participation",
    13: "ca_marks_range",
    14: "sleep_hours",
    15: "motivation",
    16: "time_management",
    17: "internet_quality",
    18: "part_time_work",
    19: "academic_stress",
    20: "travel_time",
    21: "study_resources",
    22: TARGET,
    23: "academic_support",
}


# ---------------------------------------------------------
# FINAL 16 PREDICTORS
# ---------------------------------------------------------

PREDICTORS = [
    "year_of_study",
    "degree_area",
    "study_mode",
    "attendance",
    "study_hours",
    "lms_usage",
    "assignment_habit",
    "participation",
    "sleep_hours",
    "motivation",
    "time_management",
    "internet_quality",
    "part_time_work",
    "academic_stress",
    "travel_time",
    "study_resources",
]


# ---------------------------------------------------------
# FIND RAW DATASET
# ---------------------------------------------------------

def find_raw_file():
    """
    Find one private survey dataset inside data/raw/.

    Excel is preferred because the final research workflow
    used an Excel export.
    """

    excel_files = list(RAW_FOLDER.glob("*.xlsx"))
    csv_files = list(RAW_FOLDER.glob("*.csv"))

    if excel_files:
        return excel_files[0]

    if csv_files:
        return csv_files[0]

    return None


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_dataset(file_path):
    """Load Excel or CSV survey data."""

    if file_path.suffix.lower() == ".xlsx":
        return pd.read_excel(file_path)

    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)

    raise ValueError("Unsupported file format.")


# ---------------------------------------------------------
# COLUMN CLEANING
# ---------------------------------------------------------

def rename_columns(df):
    """Convert long Google Form headings to short names."""

    rename_map = {}

    for column in df.columns:

        column_name = str(column).strip()

        if column_name.lower() == "timestamp":
            rename_map[column] = "timestamp"
            continue

        question_match = re.match(
            r"^(\d+)\.",
            column_name
        )

        if question_match:

            question_number = int(
                question_match.group(1)
            )

            rename_map[column] = QUESTION_NAMES.get(
                question_number,
                f"question_{question_number}"
            )

        else:

            safe_name = re.sub(
                r"[^a-z0-9]+",
                "_",
                column_name.lower()
            ).strip("_")

            rename_map[column] = safe_name

    return df.rename(
        columns=rename_map
    ).copy()


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text_values(df):
    """Clean spaces and common encoding problems."""

    cleaned_df = df.copy()

    for column in cleaned_df.select_dtypes(
        include=["object", "string"]
    ).columns:

        cleaned_df[column] = (
            cleaned_df[column]
            .astype("string")
            .str.strip()
            .str.replace(
                "â€“",
                "–",
                regex=False
            )
            .str.replace(
                "â€”",
                "—",
                regex=False
            )
        )

    return cleaned_df


# ---------------------------------------------------------
# TARGET ENCODING
# ---------------------------------------------------------

def encode_target(value):
    """Encode Low, Average and High as 0, 1 and 2."""

    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()

    if text.startswith("low"):
        return 0

    if text.startswith("average"):
        return 1

    if text.startswith("high"):
        return 2

    return pd.NA


# ---------------------------------------------------------
# Q13 CONSISTENCY CHECK
# ---------------------------------------------------------

def expected_class_from_ca(value):
    """
    Convert CA-mark category to a comparable class.

    Q13 is used ONLY as a quality check.
    It is not used as a model predictor and does not replace
    the student's Q22 target response.
    """

    if pd.isna(value):
        return None

    text = (
        str(value)
        .strip()
        .lower()
        .replace("–", "-")
    )

    if "below 40" in text:
        return 0

    if "40-54" in text or "55-69" in text:
        return 1

    if "70" in text and "above" in text:
        return 2

    return None


def check_q13_q22_consistency(df):
    """Compare Q13 with Q22 without changing Q22."""

    result = {
        "consistent": 0,
        "inconsistent": 0,
        "not_sure": 0,
    }

    if "ca_marks_range" not in df.columns:
        return result

    for _, row in df.iterrows():

        expected = expected_class_from_ca(
            row["ca_marks_range"]
        )

        actual = encode_target(
            row[TARGET]
        )

        if expected is None or pd.isna(actual):
            result["not_sure"] += 1

        elif expected == actual:
            result["consistent"] += 1

        else:
            result["inconsistent"] += 1

    return result


# ---------------------------------------------------------
# MAIN PREPROCESSING
# ---------------------------------------------------------

def main():

    raw_file = find_raw_file()

    if raw_file is None:

        print("No survey dataset found.")
        print("Place the private .xlsx or .csv file inside:")
        print("data/raw/")

        return

    try:

        print("Loading:", raw_file)

        # Preserve original imported dataframe
        df_raw = load_dataset(raw_file)

        # Create working copy
        df = df_raw.copy()

        df = rename_columns(df)
        df = clean_text_values(df)

        raw_rows = len(df)
        raw_columns = len(df.columns)

        raw_missing = int(
            df.isna().sum().sum()
        )

        print("\nBefore screening")
        print("Rows:", raw_rows)
        print("Columns:", raw_columns)
        print("Missing values:", raw_missing)

        # -------------------------------------------------
        # EXACT DUPLICATE CHECK
        # -------------------------------------------------

        duplicate_columns = [
            column
            for column in df.columns
            if column != "timestamp"
        ]

        duplicate_mask = df.duplicated(
            subset=duplicate_columns,
            keep="first"
        )

        duplicate_count = int(
            duplicate_mask.sum()
        )

        df = df.loc[
            ~duplicate_mask
        ].copy()

        # -------------------------------------------------
        # Q13-Q22 QUALITY CHECK
        # -------------------------------------------------

        consistency = (
            check_q13_q22_consistency(df)
        )

        # -------------------------------------------------
        # ELIGIBILITY SCREENING
        # -------------------------------------------------

        consent_mask = (
            df["consent"]
            .fillna("")
            .astype(str)
            .str.contains(
                "agree",
                case=False
            )
        )

        studying_mask = (
            df["currently_studying"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq("yes")
        )

        private_mask = (
            df["institution_type"]
            .fillna("")
            .astype(str)
            .str.contains(
                "private",
                case=False
            )
        )

        consent_count = int(
            consent_mask.sum()
        )

        studying_count = int(
            studying_mask.sum()
        )

        private_count = int(
            private_mask.sum()
        )

        basic_eligibility_mask = (
            consent_mask
            & studying_mask
            & private_mask
        )

        basic_eligible_count = int(
            basic_eligibility_mask.sum()
        )

        df = df.loc[
            basic_eligibility_mask
        ].copy()

        # -------------------------------------------------
        # TARGET
        # -------------------------------------------------

        df[TARGET] = df[TARGET].apply(
            encode_target
        )

        invalid_target_count = int(
            df[TARGET].isna().sum()
        )

        df = df.dropna(
            subset=[TARGET]
        ).copy()

        df[TARGET] = (
            df[TARGET]
            .astype(int)
        )

        # -------------------------------------------------
        # CHECK REQUIRED VARIABLES
        # -------------------------------------------------

        missing_columns = [
            column
            for column in PREDICTORS + [TARGET]
            if column not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        # -------------------------------------------------
        # FINAL MODELLING DATASET
        # -------------------------------------------------

        cleaned_df = df[
            PREDICTORS + [TARGET]
        ].copy()

        final_eligible_count = len(
            cleaned_df
        )

        excluded_count = (
            raw_rows
            - final_eligible_count
        )

        # -------------------------------------------------
        # SAVE PRIVATE PROCESSED DATA
        # -------------------------------------------------

        PROCESSED_FOLDER.mkdir(
            parents=True,
            exist_ok=True
        )

        cleaned_df.to_csv(
            OUTPUT_FILE,
            index=False
        )

        # -------------------------------------------------
        # TARGET DISTRIBUTION
        # -------------------------------------------------

        target_distribution = (
            cleaned_df[TARGET]
            .value_counts()
            .sort_index()
        )

        # -------------------------------------------------
        # REPORT
        # -------------------------------------------------

        report_lines = [

            "FINAL PREPROCESSING REPORT",
            "=" * 50,
            "",

            f"Raw responses: {raw_rows}",
            f"Raw columns: {raw_columns}",
            f"Raw missing values: {raw_missing}",
            f"Exact duplicate responses detected: {duplicate_count}",

            "",

            "SCREENING COUNTS",
            "-" * 50,

            f"Consent eligible: {consent_count}",
            f"Currently studying - Yes: {studying_count}",
            f"Private institution: {private_count}",
            f"Combined basic eligibility: {basic_eligible_count}",
            f"Invalid target after eligibility screening: {invalid_target_count}",
            f"Final eligible records: {final_eligible_count}",
            f"Total excluded records: {excluded_count}",

            "",

            (
                "Note: individual screening counts may overlap "
                "and should not be subtracted sequentially."
            ),

            "",

            "Q13 AND Q22 CONSISTENCY CHECK",
            "-" * 50,

            f"Consistent: {consistency['consistent']}",
            f"Inconsistent: {consistency['inconsistent']}",
            f"Not sure/uncheckable: {consistency['not_sure']}",

            "",

            (
                "Q13 was used only as a consistency check. "
                "It was not included as a predictor and was "
                "not used to overwrite the Q22 target."
            ),

            "",

            "FINAL TARGET DISTRIBUTION",
            "-" * 50,

            "0 = Low",
            "1 = Average",
            "2 = High",

            target_distribution.to_string(),

            "",

            f"Number of predictors: {len(PREDICTORS)}",

            (
                "Remaining missing predictor values: "
                f"{int(cleaned_df[PREDICTORS].isna().sum().sum())}"
            ),

            "",

            (
                "Imputation, encoding and model-specific scaling "
                "are performed inside the machine-learning pipelines."
            ),

        ]

        REPORT_FILE.write_text(
            "\n".join(report_lines),
            encoding="utf-8"
        )

        # -------------------------------------------------
        # FINAL OUTPUT
        # -------------------------------------------------

        print("\nPreprocessing completed.")
        print(
            "Final eligible records:",
            final_eligible_count
        )

        print(
            "Number of predictors:",
            len(PREDICTORS)
        )

        print(
            "\nTarget distribution:"
        )

        print(
            target_distribution
        )

        print(
            "\nPrivate cleaned dataset saved to:",
            OUTPUT_FILE
        )

        print(
            "Preprocessing report saved to:",
            REPORT_FILE
        )

    except (
        OSError,
        ValueError,
        KeyError,
        pd.errors.ParserError,
    ) as error:

        print("\nPreprocessing failed.")
        print("Reason:", error)


if __name__ == "__main__":
    main()

"""
Final model-training workflow for the research project:

A Comparative Study of Academic Performance Prediction Among
Sri Lankan Private Campus Students Using Survey-Based Factors.

Models:
1. Logistic Regression
2. Decision Tree
3. Random Forest

Validation:
- Stratified 5-fold outer cross-validation
- Stratified 3-fold inner GridSearchCV
- Macro F1 as the primary tuning metric
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATA_FILE = Path(
    "data/processed/cleaned_survey_data.csv"
)

RESULTS_FOLDER = Path("results/tables")

FOLD_RESULTS_FILE = (
    RESULTS_FOLDER / "nested_cv_fold_results.csv"
)

FINAL_RESULTS_FILE = (
    RESULTS_FOLDER / "final_model_results.csv"
)

PARAMETERS_FILE = (
    RESULTS_FOLDER / "nested_cv_best_parameters.json"
)

TARGET_COLUMN = "academic_performance_level"

RANDOM_STATE = 42


# ---------------------------------------------------------
# FEATURE GROUPS
# ---------------------------------------------------------

# Unordered categorical predictors
NOMINAL_FEATURES = [
    "degree_area",
    "study_mode",
]

# Ordered categorical predictors
ORDINAL_FEATURES = [
    "year_of_study",
    "attendance",
    "study_hours",
    "lms_usage",
    "assignment_habit",
    "participation",
    "sleep_hours",
    "motivation",
    "time_management",
    "internet_quality",
    "academic_stress",
    "travel_time",
    "study_resources",
]

# Binary predictor
BINARY_FEATURES = [
    "part_time_work",
]

ALL_FEATURES = (
    NOMINAL_FEATURES
    + ORDINAL_FEATURES
    + BINARY_FEATURES
)


# ---------------------------------------------------------
# ORDERED CATEGORIES
# ---------------------------------------------------------

ORDINAL_CATEGORIES = [

    # year_of_study
    [
        "Year 1",
        "Year 2",
        "Year 3",
        "Year 4",
    ],

    # attendance
    [
        "Less than 50%",
        "50–69%",
        "70–84%",
        "85% and above",
    ],

    # study_hours
    [
        "<1 hour",
        "1–2 hours",
        "3–4 hours",
        ">4 hours",
    ],

    # lms_usage
    [
        "Rarely",
        "Sometimes",
        "Often",
        "Very often",
    ],

    # assignment_habit
    [
        "Usually late",
        "Sometimes late",
        "Usually on time",
        "Always on time",
    ],

    # participation
    [
        "Low",
        "Medium",
        "High",
    ],

    # sleep_hours
    [
        "<5 hours",
        "5–6 hours",
        "7–8 hours",
        ">8 hours",
    ],

    # motivation
    [
        "Low",
        "Medium",
        "High",
    ],

    # time_management
    [
        "Poor",
        "Average",
        "Good",
    ],

    # internet_quality
    [
        "Poor",
        "Average",
        "Good",
        "Very good",
    ],

    # academic_stress
    [
        "Low",
        "Medium",
        "High",
    ],

    # travel_time
    [
        "<30 minutes",
        "30 minutes–1 hour",
        "1–2 hours",
        ">2 hours",
    ],

    # study_resources
    [
        "Poor",
        "Average",
        "Good",
        "Very good",
    ],
]


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():
    """Load the private cleaned modelling dataset."""

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            "Cleaned dataset was not found. "
            "Run src/preprocessing.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = (
        ALL_FEATURES
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    X = df[
        ALL_FEATURES
    ].copy()

    y = df[
        TARGET_COLUMN
    ].copy()

    if y.isna().any():

        raise ValueError(
            "Target contains missing values."
        )

    y = y.astype(int)

    valid_labels = {0, 1, 2}

    if not set(
        y.unique()
    ).issubset(valid_labels):

        raise ValueError(
            "Target must contain only 0, 1 and 2."
        )

    return X, y


# ---------------------------------------------------------
# PREPROCESSOR
# ---------------------------------------------------------

def create_preprocessor():
    """Create preprocessing for all predictor groups."""

    nominal_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[
                        ["No", "Yes"]
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "nominal",
                nominal_pipeline,
                NOMINAL_FEATURES,
            ),
            (
                "ordinal",
                ordinal_pipeline,
                ORDINAL_FEATURES,
            ),
            (
                "binary",
                binary_pipeline,
                BINARY_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ---------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------

def create_models():

    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "scaler",
                StandardScaler(
                    with_mean=False
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    decision_tree_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models = {

        "Logistic Regression": {
            "pipeline": logistic_pipeline,
            "parameters": {
                "model__C": [
                    0.1,
                    1,
                    10,
                ]
            },
        },

        "Decision Tree": {
            "pipeline": decision_tree_pipeline,
            "parameters": {
                "model__max_depth": [
                    3,
                    5,
                    8,
                    None,
                ],
                "model__min_samples_leaf": [
                    1,
                    3,
                    5,
                ],
            },
        },

        "Random Forest": {
            "pipeline": random_forest_pipeline,
            "parameters": {
                "model__n_estimators": [
                    100,
                    200,
                ],
                "model__max_depth": [
                    5,
                    10,
                    None,
                ],
                "model__min_samples_leaf": [
                    1,
                    2,
                    4,
                ],
            },
        },
    }

    return models


# ---------------------------------------------------------
# SCORING
# ---------------------------------------------------------

def create_scoring():

    return {

        "accuracy": "accuracy",

        "precision_macro": make_scorer(
            precision_score,
            average="macro",
            zero_division=0,
        ),

        "recall_macro": make_scorer(
            recall_score,
            average="macro",
            zero_division=0,
        ),

        "f1_macro": make_scorer(
            f1_score,
            average="macro",
            zero_division=0,
        ),
    }


# ---------------------------------------------------------
# JSON CONVERSION
# ---------------------------------------------------------

def convert_json_value(value: Any):

    if isinstance(
        value,
        np.integer
    ):
        return int(value)

    if isinstance(
        value,
        np.floating
    ):
        return float(value)

    if isinstance(
        value,
        np.ndarray
    ):
        return value.tolist()

    return value


# ---------------------------------------------------------
# MAIN TRAINING
# ---------------------------------------------------------

def main():

    try:

        X, y = load_data()

        print(
            "\nDataset loaded successfully."
        )

        print(
            "Eligible participants:",
            len(X)
        )

        print(
            "Predictors:",
            len(X.columns)
        )

        print(
            "\nTarget distribution:"
        )

        print(
            y.value_counts()
            .sort_index()
        )

        # -------------------------------------------------
        # CROSS-VALIDATION
        # -------------------------------------------------

        outer_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        inner_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        scoring = create_scoring()

        model_definitions = (
            create_models()
        )

        all_fold_results = []
        all_summary_results = []
        all_best_parameters = {}

        RESULTS_FOLDER.mkdir(
            parents=True,
            exist_ok=True
        )

        # -------------------------------------------------
        # TRAIN EACH MODEL
        # -------------------------------------------------

        for (
            model_name,
            definition
        ) in model_definitions.items():

            print(
                "\n"
                + "=" * 60
            )

            print(
                "Training:",
                model_name
            )

            print(
                "=" * 60
            )

            grid_search = GridSearchCV(
                estimator=definition[
                    "pipeline"
                ],
                param_grid=definition[
                    "parameters"
                ],
                scoring="f1_macro",
                cv=inner_cv,
                refit=True,
                n_jobs=-1,
                error_score="raise",
            )

            cv_results = cross_validate(
                estimator=grid_search,
                X=X,
                y=y,
                cv=outer_cv,
                scoring=scoring,
                return_estimator=True,
                n_jobs=1,
                error_score="raise",
            )

            fold_parameters = []

            # ---------------------------------------------
            # SAVE OUTER-FOLD RESULTS
            # ---------------------------------------------

            for fold_number in range(5):

                fold_result = {

                    "Model":
                        model_name,

                    "Fold":
                        fold_number + 1,

                    "Accuracy":
                        cv_results[
                            "test_accuracy"
                        ][fold_number],

                    "Macro Precision":
                        cv_results[
                            "test_precision_macro"
                        ][fold_number],

                    "Macro Recall":
                        cv_results[
                            "test_recall_macro"
                        ][fold_number],

                    "Macro F1":
                        cv_results[
                            "test_f1_macro"
                        ][fold_number],
                }

                all_fold_results.append(
                    fold_result
                )

                fitted_search = (
                    cv_results[
                        "estimator"
                    ][fold_number]
                )

                best_params = {
                    key:
                    convert_json_value(
                        value
                    )
                    for key, value
                    in fitted_search
                    .best_params_
                    .items()
                }

                fold_parameters.append(
                    {
                        "fold":
                            fold_number + 1,

                        "parameters":
                            best_params,
                    }
                )

            all_best_parameters[
                model_name
            ] = fold_parameters

            # ---------------------------------------------
            # MODEL SUMMARY
            # ---------------------------------------------

            metric_columns = {

                "Accuracy":
                    "test_accuracy",

                "Macro Precision":
                    "test_precision_macro",

                "Macro Recall":
                    "test_recall_macro",

                "Macro F1":
                    "test_f1_macro",
            }

            summary = {
                "Model":
                    model_name
            }

            for (
                display_name,
                result_key
            ) in metric_columns.items():

                values = cv_results[
                    result_key
                ]

                summary[
                    f"{display_name} Mean"
                ] = float(
                    np.mean(values)
                )

                summary[
                    f"{display_name} SD"
                ] = float(
                    np.std(
                        values,
                        ddof=1
                    )
                )

            all_summary_results.append(
                summary
            )

            print(
                "Accuracy:",
                f"{summary['Accuracy Mean']:.4f}",
                "±",
                f"{summary['Accuracy SD']:.4f}",
            )

            print(
                "Macro Precision:",
                f"{summary['Macro Precision Mean']:.4f}",
                "±",
                f"{summary['Macro Precision SD']:.4f}",
            )

            print(
                "Macro Recall:",
                f"{summary['Macro Recall Mean']:.4f}",
                "±",
                f"{summary['Macro Recall SD']:.4f}",
            )

            print(
                "Macro F1:",
                f"{summary['Macro F1 Mean']:.4f}",
                "±",
                f"{summary['Macro F1 SD']:.4f}",
            )

        # -------------------------------------------------
        # SAVE FINAL RESULTS
        # -------------------------------------------------

        fold_results_df = pd.DataFrame(
            all_fold_results
        )

        final_results_df = pd.DataFrame(
            all_summary_results
        )

        fold_results_df.to_csv(
            FOLD_RESULTS_FILE,
            index=False
        )

        final_results_df.to_csv(
            FINAL_RESULTS_FILE,
            index=False
        )

        PARAMETERS_FILE.write_text(
            json.dumps(
                all_best_parameters,
                indent=4
            ),
            encoding="utf-8"
        )

        # -------------------------------------------------
        # DISPLAY OVERALL RESULT
        # -------------------------------------------------

        best_model_row = (
            final_results_df.loc[
                final_results_df[
                    "Macro F1 Mean"
                ].idxmax()
            ]
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "Nested cross-validation completed."
        )

        print(
            "=" * 60
        )

        print(
            "\nHighest mean Macro F1:",
            best_model_row[
                "Model"
            ]
        )

        print(
            "Mean Macro F1:",
            round(
                float(
                    best_model_row[
                        "Macro F1 Mean"
                    ]
                ),
                4
            )
        )

        print(
            "\nImportant:"
        )

        print(
            "The model with the highest mean "
            "Macro F1 is not automatically best "
            "for every practical objective."
        )

        print(
            "Class-wise recall and statistical "
            "results must also be considered."
        )

        print(
            "\nSaved fold results to:"
        )

        print(
            FOLD_RESULTS_FILE
        )

        print(
            "\nSaved final model results to:"
        )

        print(
            FINAL_RESULTS_FILE
        )

        print(
            "\nSaved fold-level best parameters to:"
        )

        print(
            PARAMETERS_FILE
        )

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        OSError,
        pd.errors.ParserError,
    ) as error:

        print(
            "\nModel training failed."
        )

        print(
            "Reason:",
            error
        )


if __name__ == "__main__":
    main()

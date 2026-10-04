"""
Final model-evaluation workflow for:

A Comparative Study of Academic Performance Prediction Among
Sri Lankan Private Campus Students Using Survey-Based Factors.

This script:

1. Generates nested out-of-fold predictions
2. Calculates overall classification metrics
3. Calculates fold-level class recall
4. Creates final confusion matrices
5. Runs the Friedman statistical test
6. Runs pairwise Wilcoxon tests only if required
7. Applies Holm adjustment without an additional dependency

Participant-level predictions are NOT saved publicly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    friedmanchisquare,
    wilcoxon,
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)

from train_models import (
    RANDOM_STATE,
    create_models,
    load_data,
)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

TABLES_FOLDER = Path(
    "results/tables"
)

FIGURES_FOLDER = Path(
    "results/figures"
)

FOLD_RESULTS_FILE = (
    TABLES_FOLDER
    / "nested_cv_fold_results.csv"
)

EVALUATION_FILE = (
    TABLES_FOLDER
    / "final_model_results.csv"
)

CLASS_RECALL_FILE = (
    TABLES_FOLDER
    / "final_class_recall_table.csv"
)

STATISTICAL_FILE = (
    TABLES_FOLDER
    / "statistical_summary.csv"
)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

MODEL_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
]

CLASS_LABELS = [
    0,
    1,
    2,
]

CLASS_NAMES = [
    "Low",
    "Average",
    "High",
]

ALPHA = 0.05


# ---------------------------------------------------------
# FILE NAME HELPER
# ---------------------------------------------------------

def safe_filename(model_name):
    """Convert model name to a safe file name."""

    return (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

def save_confusion_matrix(
    model_name,
    matrix,
):
    """Create publication-ready confusion matrix PNG."""

    safe_name = safe_filename(
        model_name
    )

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    image = axis.imshow(
        matrix
    )

    figure.colorbar(
        image,
        ax=axis
    )

    axis.set_title(
        f"Tuned {model_name} Confusion Matrix"
    )

    axis.set_xlabel(
        "Predicted Class"
    )

    axis.set_ylabel(
        "Actual Class"
    )

    axis.set_xticks(
        range(
            len(CLASS_NAMES)
        )
    )

    axis.set_xticklabels(
        CLASS_NAMES
    )

    axis.set_yticks(
        range(
            len(CLASS_NAMES)
        )
    )

    axis.set_yticklabels(
        CLASS_NAMES
    )

    for row in range(
        matrix.shape[0]
    ):

        for column in range(
            matrix.shape[1]
        ):

            axis.text(
                column,
                row,
                str(
                    matrix[
                        row,
                        column
                    ]
                ),
                ha="center",
                va="center",
            )

    figure.tight_layout()

    output_file = (
        FIGURES_FOLDER
        / (
            f"tuned_{safe_name}"
            "_confusion_matrix.png"
        )
    )

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ---------------------------------------------------------
# HOLM ADJUSTMENT
# ---------------------------------------------------------

def holm_adjust(p_values):
    """
    Apply Holm correction without statsmodels.

    Returns adjusted p-values in the original order.
    """

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    number_of_tests = len(
        p_values
    )

    order = np.argsort(
        p_values
    )

    sorted_p = p_values[
        order
    ]

    adjusted_sorted = np.empty(
        number_of_tests
    )

    previous = 0.0

    for index, p_value in enumerate(
        sorted_p
    ):

        adjusted = (
            number_of_tests
            - index
        ) * p_value

        adjusted = max(
            adjusted,
            previous
        )

        adjusted = min(
            adjusted,
            1.0
        )

        adjusted_sorted[
            index
        ] = adjusted

        previous = adjusted

    adjusted_original = np.empty(
        number_of_tests
    )

    adjusted_original[
        order
    ] = adjusted_sorted

    return adjusted_original


# ---------------------------------------------------------
# SAFE WILCOXON
# ---------------------------------------------------------

def safe_wilcoxon(
    first_scores,
    second_scores,
):

    try:

        statistic, p_value = (
            wilcoxon(
                first_scores,
                second_scores,
                alternative="two-sided",
            )
        )

    except ValueError:

        statistic = 0.0
        p_value = 1.0

    return (
        float(statistic),
        float(p_value),
    )


# ---------------------------------------------------------
# NESTED EVALUATION
# ---------------------------------------------------------

def evaluate_models():
    """
    Run nested 5-fold outer evaluation.

    Hyperparameter tuning is performed inside each
    outer training fold using 3-fold GridSearchCV.
    """

    X, y = load_data()

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

    model_definitions = (
        create_models()
    )

    all_fold_results = []
    all_class_recall = []
    final_summary = []

    for model_name in MODEL_ORDER:

        print(
            "\n"
            + "=" * 60
        )

        print(
            "Evaluating:",
            model_name
        )

        print(
            "=" * 60
        )

        definition = (
            model_definitions[
                model_name
            ]
        )

        # Stores all out-of-fold
        # actual and predicted labels.
        #
        # These remain in memory only.
        all_actual = []
        all_predictions = []

        model_fold_metrics = []

        model_class_recalls = {
            "Low": [],
            "Average": [],
            "High": [],
        }

        for fold_number, (
            train_index,
            test_index,
        ) in enumerate(
            outer_cv.split(
                X,
                y
            ),
            start=1
        ):

            X_train = X.iloc[
                train_index
            ]

            X_test = X.iloc[
                test_index
            ]

            y_train = y.iloc[
                train_index
            ]

            y_test = y.iloc[
                test_index
            ]

            grid_search = GridSearchCV(
                estimator=definition[
                    "pipeline"
                ],
                param_grid=definition[
                    "parameters"
                ],
                scoring="f1_macro",
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )

            grid_search.fit(
                X_train,
                y_train
            )

            predictions = (
                grid_search.predict(
                    X_test
                )
            )

            accuracy = (
                accuracy_score(
                    y_test,
                    predictions
                )
            )

            macro_precision = (
                precision_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            )

            macro_recall = (
                recall_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            )

            macro_f1 = (
                f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            )

            fold_row = {
                "Model":
                    model_name,

                "Fold":
                    fold_number,

                "Accuracy":
                    accuracy,

                "Macro Precision":
                    macro_precision,

                "Macro Recall":
                    macro_recall,

                "Macro F1":
                    macro_f1,
            }

            all_fold_results.append(
                fold_row
            )

            model_fold_metrics.append(
                fold_row
            )

            # ---------------------------------------------
            # CLASS RECALL FOR THIS OUTER FOLD
            # ---------------------------------------------

            fold_recalls = (
                recall_score(
                    y_test,
                    predictions,
                    labels=CLASS_LABELS,
                    average=None,
                    zero_division=0,
                )
            )

            for (
                class_name,
                class_recall
            ) in zip(
                CLASS_NAMES,
                fold_recalls
            ):

                model_class_recalls[
                    class_name
                ].append(
                    float(
                        class_recall
                    )
                )

            # Keep only in memory.
            all_actual.extend(
                y_test.tolist()
            )

            all_predictions.extend(
                predictions.tolist()
            )

        # -------------------------------------------------
        # CONFUSION MATRIX
        # -------------------------------------------------

        matrix = confusion_matrix(
            all_actual,
            all_predictions,
            labels=CLASS_LABELS,
        )

        save_confusion_matrix(
            model_name,
            matrix
        )

        # -------------------------------------------------
        # CLASS-WISE RECALL SUMMARY
        # -------------------------------------------------

        for class_name in CLASS_NAMES:

            values = np.array(
                model_class_recalls[
                    class_name
                ],
                dtype=float
            )

            all_class_recall.append(
                {
                    "Model":
                        model_name,

                    "Class":
                        class_name,

                    "Recall Mean":
                        float(
                            np.mean(
                                values
                            )
                        ),

                    "Recall SD":
                        float(
                            np.std(
                                values,
                                ddof=1
                            )
                        ),
                }
            )

        # -------------------------------------------------
        # FINAL MODEL SUMMARY
        # -------------------------------------------------

        model_fold_df = (
            pd.DataFrame(
                model_fold_metrics
            )
        )

        summary = {
            "Model":
                model_name
        }

        for metric in [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
        ]:

            values = (
                model_fold_df[
                    metric
                ]
                .to_numpy(
                    dtype=float
                )
            )

            summary[
                f"{metric} Mean"
            ] = float(
                np.mean(
                    values
                )
            )

            summary[
                f"{metric} SD"
            ] = float(
                np.std(
                    values,
                    ddof=1
                )
            )

        final_summary.append(
            summary
        )

        print(
            "Mean Macro F1:",
            round(
                summary[
                    "Macro F1 Mean"
                ],
                4
            )
        )

    # -----------------------------------------------------
    # SAVE AGGREGATED RESULTS
    # -----------------------------------------------------

    fold_results_df = (
        pd.DataFrame(
            all_fold_results
        )
    )

    class_recall_df = (
        pd.DataFrame(
            all_class_recall
        )
    )

    final_results_df = (
        pd.DataFrame(
            final_summary
        )
    )

    fold_results_df.to_csv(
        FOLD_RESULTS_FILE,
        index=False
    )

    class_recall_df.to_csv(
        CLASS_RECALL_FILE,
        index=False
    )

    final_results_df.to_csv(
        EVALUATION_FILE,
        index=False
    )

    return fold_results_df


# ---------------------------------------------------------
# STATISTICAL TESTING
# ---------------------------------------------------------

def run_statistical_tests(
    fold_results_df
):
    """
    Compare fold-level Macro F1 scores.

    Friedman test is performed first.

    Wilcoxon pairwise tests are only performed
    when Friedman is statistically significant.
    """

    score_table = (
        fold_results_df
        .pivot(
            index="Fold",
            columns="Model",
            values="Macro F1",
        )
    )

    score_table = (
        score_table[
            MODEL_ORDER
        ]
        .dropna()
    )

    friedman_statistic, friedman_p = (
        friedmanchisquare(
            score_table[
                "Logistic Regression"
            ],
            score_table[
                "Decision Tree"
            ],
            score_table[
                "Random Forest"
            ],
        )
    )

    statistical_rows = [
        {
            "Comparison":
                "All three models",

            "Statistical Test":
                "Friedman test",

            "Statistic":
                float(
                    friedman_statistic
                ),

            "Raw p-value":
                float(
                    friedman_p
                ),

            "Adjusted p-value":
                float(
                    friedman_p
                ),

            "Decision":
                (
                    "Reject null hypothesis"
                    if friedman_p < ALPHA
                    else
                    "Fail to reject null hypothesis"
                ),
        }
    ]

    print(
        "\nFriedman statistic:",
        round(
            float(
                friedman_statistic
            ),
            4
        )
    )

    print(
        "Friedman p-value:",
        round(
            float(
                friedman_p
            ),
            4
        )
    )

    pairwise_comparisons = [
        (
            "Logistic Regression",
            "Decision Tree",
        ),
        (
            "Logistic Regression",
            "Random Forest",
        ),
        (
            "Decision Tree",
            "Random Forest",
        ),
    ]

    # -----------------------------------------------------
    # PAIRWISE TESTS ONLY IF FRIEDMAN IS SIGNIFICANT
    # -----------------------------------------------------

    if friedman_p < ALPHA:

        raw_p_values = []
        pairwise_results = []

        for (
            first_model,
            second_model
        ) in pairwise_comparisons:

            statistic, raw_p = (
                safe_wilcoxon(
                    score_table[
                        first_model
                    ],
                    score_table[
                        second_model
                    ],
                )
            )

            raw_p_values.append(
                raw_p
            )

            pairwise_results.append(
                {
                    "first":
                        first_model,

                    "second":
                        second_model,

                    "statistic":
                        statistic,

                    "raw_p":
                        raw_p,
                }
            )

        adjusted_values = (
            holm_adjust(
                raw_p_values
            )
        )

        for (
            result,
            adjusted_p
        ) in zip(
            pairwise_results,
            adjusted_values
        ):

            statistical_rows.append(
                {
                    "Comparison":
                        (
                            f"{result['first']} "
                            f"vs {result['second']}"
                        ),

                    "Statistical Test":
                        (
                            "Wilcoxon signed-rank "
                            "test with Holm adjustment"
                        ),

                    "Statistic":
                        result[
                            "statistic"
                        ],

                    "Raw p-value":
                        result[
                            "raw_p"
                        ],

                    "Adjusted p-value":
                        float(
                            adjusted_p
                        ),

                    "Decision":
                        (
                            "Statistically significant"
                            if adjusted_p < ALPHA
                            else
                            "Not statistically significant"
                        ),
                }
            )

    else:

        statistical_rows.append(
            {
                "Comparison":
                    "Pairwise comparisons",

                "Statistical Test":
                    "Wilcoxon tests not performed",

                "Statistic":
                    np.nan,

                "Raw p-value":
                    np.nan,

                "Adjusted p-value":
                    np.nan,

                "Decision":
                    (
                        "Friedman test was "
                        "not statistically significant"
                    ),
            }
        )

    statistical_df = (
        pd.DataFrame(
            statistical_rows
        )
    )

    statistical_df.to_csv(
        STATISTICAL_FILE,
        index=False
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    try:

        TABLES_FOLDER.mkdir(
            parents=True,
            exist_ok=True
        )

        FIGURES_FOLDER.mkdir(
            parents=True,
            exist_ok=True
        )

        fold_results_df = (
            evaluate_models()
        )

        run_statistical_tests(
            fold_results_df
        )

        print(
            "\nEvaluation completed successfully."
        )

        print(
            "\nSaved final model results to:"
        )

        print(
            EVALUATION_FILE
        )

        print(
            "\nSaved class-wise recall to:"
        )

        print(
            CLASS_RECALL_FILE
        )

        print(
            "\nSaved statistical results to:"
        )

        print(
            STATISTICAL_FILE
        )

        print(
            "\nConfusion matrices saved to:"
        )

        print(
            FIGURES_FOLDER
        )

        print(
            "\nParticipant-level predictions "
            "were not saved."
        )

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        OSError,
        pd.errors.ParserError,
    ) as error:

        print(
            "\nEvaluation failed."
        )

        print(
            "Reason:",
            error
        )


if __name__ == "__main__":
    main()

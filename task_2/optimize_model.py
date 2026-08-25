from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from main import RANDOM_STATE, RESULTS_DIR, load_dataset, split_dataset, validate_dataset


GRID_SEARCH_RESULTS_FILE = RESULTS_DIR / "grid_search_results.csv"
OPTIMIZED_METRICS_FILE = RESULTS_DIR / "optimized_metrics.csv"
OPTIMIZED_CLASSIFICATION_REPORT_FILE = (
    RESULTS_DIR / "optimized_classification_report.csv"
)
OPTIMIZED_CONFUSION_MATRIX_FILE = RESULTS_DIR / "optimized_confusion_matrix.png"
MODEL_COMPARISON_FILE = RESULTS_DIR / "model_comparison.csv"
PARAMETER_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__sublinear_tf": [False, True],
    "classifier__C": [0.5, 1.0, 2.0, 4.0],
}


def build_optimization_pipeline() -> Pipeline:
    """Build the baseline model family with a higher convergence ceiling."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=False)),
            (
                "classifier",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )


def count_parameter_combinations() -> int:
    """Calculate the number of GridSearchCV candidates from the configured grid."""
    combination_count = 1
    for values in PARAMETER_GRID.values():
        combination_count *= len(values)
    return combination_count


def optimize_model(
    training_features: pd.Series, training_target: pd.Series
) -> GridSearchCV:
    """Select hyperparameters using only stratified training-data folds."""
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    grid_search = GridSearchCV(
        estimator=build_optimization_pipeline(),
        param_grid=PARAMETER_GRID,
        scoring="f1",
        cv=cross_validation,
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search.fit(training_features, training_target)

    candidate_count = count_parameter_combinations()
    print("\n" + "=" * 60)
    print("GRID SEARCH OPTIMIZATION")
    print("=" * 60)
    print(f"Candidate combinations: {candidate_count}")
    print(f"Cross-validation folds: {cross_validation.n_splits}")
    print(f"Optimization fits: {candidate_count * cross_validation.n_splits}")
    print("Optimization metric: F1-score")
    print(f"\nBest cross-validation F1: {grid_search.best_score_:.4f}")
    print("\nBest parameters:")
    for parameter, value in grid_search.best_params_.items():
        print(f"{parameter}: {value}")
    print("=" * 60)
    return grid_search


def evaluate_optimized_model(
    optimized_model: Pipeline,
    testing_features: pd.Series,
    testing_target: pd.Series,
    best_cv_f1: float,
) -> dict[str, float | int]:
    """Evaluate the selected estimator once on the independent test partition."""
    predictions = optimized_model.predict(testing_features)
    matrix = confusion_matrix(testing_target, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    metrics = {
        "accuracy": accuracy_score(testing_target, predictions),
        "precision": precision_score(testing_target, predictions),
        "recall": recall_score(testing_target, predictions),
        "f1_score": f1_score(testing_target, predictions),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "test_observations": len(testing_target),
        "best_cv_f1": best_cv_f1,
    }
    report = classification_report(
        testing_target,
        predictions,
        labels=[0, 1],
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics.items(), columns=["metric", "value"]).to_csv(
        OPTIMIZED_METRICS_FILE, index=False
    )
    pd.DataFrame(report).transpose().to_csv(OPTIMIZED_CLASSIFICATION_REPORT_FILE)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Negative", "Positive"],
    )
    figure, axes = plt.subplots(figsize=(6, 5))
    display.plot(ax=axes, values_format="d")
    axes.set_title("Optimized Sentiment Model Confusion Matrix")
    figure.tight_layout()
    figure.savefig(OPTIMIZED_CONFUSION_MATRIX_FILE, dpi=200)
    plt.close(figure)

    print("\n" + "=" * 60)
    print("OPTIMIZED MODEL EVALUATION")
    print("=" * 60)
    print(f"Test observations: {len(testing_target)}")
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix:\n{matrix}")
    print(f"\nTrue negatives: {true_negative}")
    print(f"False positives: {false_positive}")
    print(f"False negatives: {false_negative}")
    print(f"True positives: {true_positive}")
    print("=" * 60)
    return metrics


def save_model_comparison(optimized_metrics: dict[str, float | int]) -> None:
    """Compare persisted baseline metrics against the optimized holdout results."""
    baseline_metrics = pd.read_csv(RESULTS_DIR / "baseline_metrics.csv").set_index(
        "metric"
    )["value"]
    comparison_rows = []
    for metric in ("accuracy", "precision", "recall", "f1_score"):
        baseline_value = float(baseline_metrics[metric])
        optimized_value = float(optimized_metrics[metric])
        comparison_rows.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "optimized": optimized_value,
                "absolute_change": optimized_value - baseline_value,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(MODEL_COMPARISON_FILE, index=False)

    print("\n" + "=" * 60)
    print("BASELINE VS OPTIMIZED MODEL")
    print("=" * 60)
    print(f"{'Metric':<12}{'Baseline':>12}{'Optimized':>12}{'Change':>12}")
    for row in comparison.itertuples(index=False):
        print(
            f"{row.metric:<12}{row.baseline:>12.4f}{row.optimized:>12.4f}"
            f"{row.absolute_change:>+12.4f}"
        )
    print("=" * 60)


def main() -> None:
    dataset = load_dataset()
    validate_dataset(dataset)
    training_features, testing_features, training_target, testing_target = split_dataset(
        dataset
    )
    grid_search = optimize_model(training_features, training_target)
    optimized_metrics = evaluate_optimized_model(
        grid_search.best_estimator_,
        testing_features,
        testing_target,
        grid_search.best_score_,
    )
    pd.DataFrame(grid_search.cv_results_).to_csv(GRID_SEARCH_RESULTS_FILE, index=False)
    save_model_comparison(optimized_metrics)


if __name__ == "__main__":
    main()
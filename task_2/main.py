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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "restaurant_reviews_processed.csv"
RESULTS_DIR = BASE_DIR / "results"
METRICS_FILE = RESULTS_DIR / "baseline_metrics.csv"
CLASSIFICATION_REPORT_FILE = RESULTS_DIR / "baseline_classification_report.csv"
CONFUSION_MATRIX_FILE = RESULTS_DIR / "baseline_confusion_matrix.png"
RANDOM_STATE = 42
TEST_SIZE = 0.20
REQUIRED_COLUMNS = {
    "original_review",
    "cleaned_review",
    "tokens",
    "sentiment",
}


def load_dataset() -> pd.DataFrame:
    """Load the processed Task 2 restaurant review dataset."""
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    return pd.read_csv(DATA_FILE)


def validate_dataset(dataset: pd.DataFrame) -> None:
    """Validate the required structure and labels before modeling begins."""
    missing_columns = REQUIRED_COLUMNS.difference(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    if len(dataset) != 1_000:
        raise ValueError(f"Expected 1000 rows; found {len(dataset)}.")

    sentiment_labels = sorted(dataset["sentiment"].dropna().unique().tolist())
    if sentiment_labels != [0, 1]:
        raise ValueError(
            f"Expected sentiment labels [0, 1]; found {sentiment_labels}."
        )

    sentiment_counts = dataset["sentiment"].value_counts()
    positive_reviews = int(sentiment_counts.get(1, 0))
    negative_reviews = int(sentiment_counts.get(0, 0))
    if positive_reviews != 500 or negative_reviews != 500:
        raise ValueError(
            "Expected 500 positive and 500 negative reviews; "
            f"found {positive_reviews} positive and {negative_reviews} negative."
        )

    missing_cleaned_reviews = int(dataset["cleaned_review"].isna().sum())
    missing_sentiment_labels = int(dataset["sentiment"].isna().sum())
    if missing_cleaned_reviews or missing_sentiment_labels:
        raise ValueError(
            "Required model-input values are missing: "
            f"cleaned_review={missing_cleaned_reviews}, "
            f"sentiment={missing_sentiment_labels}."
        )

    print("=" * 60)
    print("D803 TASK 2 DATASET VALIDATION")
    print("=" * 60)
    print(f"Dataset: {DATA_FILE.name}")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"Positive reviews: {positive_reviews}")
    print(f"Negative reviews: {negative_reviews}")
    print(f"Missing cleaned reviews: {missing_cleaned_reviews}")
    print(f"Missing sentiment labels: {missing_sentiment_labels}")
    print(f"Sentiment labels: {sentiment_labels}")
    print("Validation status: PASSED")
    print("=" * 60)


def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create a reproducible, stratified split for model development."""
    features = dataset["cleaned_review"]
    target = dataset["sentiment"]
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def build_baseline_model() -> Pipeline:
    """Build the baseline TF-IDF and Logistic Regression pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=False)),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def train_model(
    model: Pipeline, training_features: pd.Series, training_target: pd.Series
) -> int:
    """Fit the baseline model using only the training partition."""
    model.fit(training_features, training_target)
    feature_count = len(model.named_steps["tfidf"].get_feature_names_out())
    print("Baseline model training: COMPLETE")
    print(f"TF-IDF features: {feature_count}")
    return feature_count


def demonstrate_predictions(model: Pipeline) -> None:
    """Display inference from the trained baseline model."""
    sample_texts = [
        "food excellent service friendly",
        "food terrible service slow",
        "not good restaurant",
    ]
    predicted_labels = model.predict(sample_texts)

    print("\nBASELINE MODEL SAMPLE PREDICTIONS")
    for sample_text, predicted_label in zip(sample_texts, predicted_labels):
        sentiment_name = "positive" if predicted_label == 1 else "negative"
        print(f"\nText: {sample_text}")
        print(f"Prediction: {sentiment_name}")


def evaluate_model(
    model: Pipeline, testing_features: pd.Series, testing_target: pd.Series
) -> dict[str, float | int]:
    """Evaluate the trained baseline model on the independent test partition."""
    predictions = model.predict(testing_features)
    accuracy = accuracy_score(testing_target, predictions)
    precision = precision_score(testing_target, predictions)
    recall = recall_score(testing_target, predictions)
    f1 = f1_score(testing_target, predictions)
    matrix = confusion_matrix(testing_target, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    report = classification_report(
        testing_target,
        predictions,
        labels=[0, 1],
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "test_observations": len(testing_target),
    }
    pd.DataFrame(metrics.items(), columns=["metric", "value"]).to_csv(
        METRICS_FILE, index=False
    )
    pd.DataFrame(report).transpose().to_csv(CLASSIFICATION_REPORT_FILE)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Negative", "Positive"],
    )
    figure, axes = plt.subplots(figsize=(6, 5))
    display.plot(ax=axes, values_format="d")
    axes.set_title("Baseline Sentiment Model Confusion Matrix")
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_FILE, dpi=200)
    plt.close(figure)

    print("\n" + "=" * 60)
    print("BASELINE MODEL EVALUATION")
    print("=" * 60)
    print(f"Test observations: {len(testing_target)}")
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"\nConfusion Matrix:\n{matrix}")
    print(f"\nTrue negatives: {true_negative}")
    print(f"False positives: {false_positive}")
    print(f"False negatives: {false_negative}")
    print(f"True positives: {true_positive}")
    print("\nClassification Report:")
    print(
        classification_report(
            testing_target,
            predictions,
            target_names=["negative", "positive"],
            digits=4,
            zero_division=0,
        )
    )
    print("=" * 60)
    return metrics


def print_model_summary(
    training_features: pd.Series,
    testing_features: pd.Series,
    training_target: pd.Series,
    testing_target: pd.Series,
    feature_count: int,
) -> None:
    """Print the baseline model configuration and split summary."""
    training_counts = training_target.value_counts()
    testing_counts = testing_target.value_counts()

    print("\n" + "=" * 60)
    print("D803 TASK 2 BASELINE NLP MODEL")
    print("=" * 60)
    print("Feature: cleaned_review")
    print("Target: sentiment")
    print(f"\nTraining observations: {len(training_features)}")
    print(f"Testing observations: {len(testing_features)}")
    print(f"\nTraining positive: {int(training_counts.get(1, 0))}")
    print(f"Training negative: {int(training_counts.get(0, 0))}")
    print(f"Testing positive: {int(testing_counts.get(1, 0))}")
    print(f"Testing negative: {int(testing_counts.get(0, 0))}")
    print("\nText representation: TF-IDF")
    print("Classifier: Logistic Regression")
    print(f"TF-IDF features: {feature_count}")
    print(f"Random state: {RANDOM_STATE}")
    print("\nModel training status: PASSED")
    print("Requirement B working model status: COMPLETE")
    print("=" * 60)


def main() -> None:
    dataset = load_dataset()
    validate_dataset(dataset)
    training_features, testing_features, training_target, testing_target = split_dataset(
        dataset
    )
    model = build_baseline_model()
    feature_count = train_model(model, training_features, training_target)
    demonstrate_predictions(model)
    evaluate_model(model, testing_features, testing_target)
    print_model_summary(
        training_features,
        testing_features,
        training_target,
        testing_target,
        feature_count,
    )


if __name__ == "__main__":
    main()
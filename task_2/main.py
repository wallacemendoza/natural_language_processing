from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "restaurant_reviews_processed.csv"
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


def main() -> None:
    dataset = load_dataset()
    validate_dataset(dataset)
    print("\nTask 2 dataset is ready for NLP model development.")


if __name__ == "__main__":
    main()
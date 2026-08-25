# D803 Task 2 - NLP Sentiment Classification

## Objective

Task 2 builds, evaluates, and optimizes a binary sentiment classification model using the preprocessed restaurant reviews generated in Task 1.

## Dataset

`data/restaurant_reviews_processed.csv` contains 1,000 reviews: 500 positive and 500 negative. The modeling feature will be `cleaned_review`, and the target will be `sentiment`.

## Planned Modeling Workflow

1. TF-IDF feature extraction
2. Training/test splitting
3. Baseline classification
4. Performance evaluation
5. Model optimization
6. Algorithm comparison
7. Final model selection

## Baseline Model

The baseline model uses `cleaned_review` as the feature and `sentiment` as the target. It represents text with TF-IDF and classifies sentiment with Logistic Regression.

- Train/test split: 80/20
- Stratification: enabled
- Random state: 42

The TF-IDF vectorizer and Logistic Regression classifier are contained in one scikit-learn Pipeline. This ensures that text features are fitted only on the training partition and prevents data leakage into the testing partition.

## Project Structure

```text
task_2/
|-- data/
|-- models/
|-- results/
|-- main.py
|-- requirements.txt
`-- README.md
```

## Current Status

Requirement B complete: baseline TF-IDF and Logistic Regression sentiment classification model implemented and validated.
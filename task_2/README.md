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

## Baseline Evaluation

The baseline model was evaluated on the held-out test partition: accuracy 88.00%, precision 87.25%, recall 89.00%, and F1-score 88.12%. The confusion matrix and detailed evaluation outputs are stored in `results/`. The complete evaluation summary is available in [requirement_e_evaluation.md](requirement_e_evaluation.md).

## Model Optimization

GridSearchCV performed five-fold stratified cross-validation on the training partition only, using F1-score to tune `classifier__C`, `tfidf__ngram_range`, and `tfidf__sublinear_tf`. The selected parameters were `C=4.0`, unigram features, and `sublinear_tf=False`. The optimized model achieved accuracy 88.50%, precision 87.38%, recall 90.00%, and F1-score 88.67%, compared with baseline values of 88.00%, 87.25%, 89.00%, and 88.12%. The complete optimization discussion is available in [requirement_f_optimization.md](requirement_f_optimization.md).

## Libraries and Tools

The project uses Python, pandas, scikit-learn, and matplotlib, with Visual Studio Code, Git, and GitLab supporting development and version control. A detailed description of the implementation libraries and development tools is available in [requirement_c_libraries_tools.md](requirement_c_libraries_tools.md).

## Implementation Documentation

The Requirement D implementation description is documented in [requirement_d_implementation.md](requirement_d_implementation.md).

## Project Structure

```text
task_2/
|-- data/
|-- models/
|-- results/
|   |-- baseline_metrics.csv
|   |-- baseline_classification_report.csv
|   |-- baseline_confusion_matrix.png
|   |-- grid_search_results.csv
|   |-- optimized_metrics.csv
|   |-- optimized_classification_report.csv
|   |-- optimized_confusion_matrix.png
|   `-- model_comparison.csv
|-- main.py
|-- optimize_model.py
|-- requirements.txt
|-- requirement_c_libraries_tools.md
|-- requirement_d_implementation.md
|-- requirement_e_evaluation.md
|-- requirement_f_optimization.md
`-- README.md
```

## Current Status

Requirements B through F are complete. The baseline sentiment model has been implemented and evaluated, and the model has been optimized using training-only stratified cross-validation and GridSearchCV. Baseline and optimized evaluation artifacts are retained separately for comparison.
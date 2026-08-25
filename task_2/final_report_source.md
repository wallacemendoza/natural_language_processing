# D803 Natural Language Processing

## Task 2 - NLP Sentiment Classification and Optimization

Wallace Mendoza

Student ID: 011862367

Western Governors University

**Competencies:**

4141.1.2 Applies NLP for Text Analysis and Generation

4141.1.3 Optimizes Natural Language Models

## B. Working NLP Model

The working NLP system classifies sentiment in 1,000 processed restaurant reviews. It uses `cleaned_review` as the feature and `sentiment` as the target, where 0 represents negative sentiment and 1 represents positive sentiment. The reviews originate from the restaurant-review dataset obtained through Kaggle (Eldsouky, n.d.) and were processed during Task 1.

The executable baseline in `task_2/main.py` validates the dataset, creates a stratified 80/20 train/test split with `random_state=42`, and trains a scikit-learn Pipeline. The pipeline represents text with TF-IDF and applies Logistic Regression as the baseline classifier, then demonstrates inference with cleaned-style sample phrases.

## C. Libraries and Tools

| Component | Role |
| --- | --- |
| Python 3.12.10 | Programming language and runtime |
| pandas 2.3.3 | Dataset loading and validation |
| scikit-learn 1.5.2 | TF-IDF, splitting, classification, evaluation, and optimization |
| matplotlib 3.10.9 | Confusion-matrix visualization |
| Visual Studio Code | Source-code development |
| Git | Local version control |
| GitLab | Remote repository and commit history |
| PowerShell | Program execution and Git commands |

Python provides the command-line application runtime. pandas loads and validates the processed data, while scikit-learn supplies `TfidfVectorizer`, `train_test_split`, `Pipeline`, Logistic Regression, evaluation metrics, and GridSearchCV. matplotlib works with `ConfusionMatrixDisplay` to generate the baseline and optimized confusion-matrix figures. Visual Studio Code, PowerShell, Git, and GitLab support development, execution, traceability, and remote history.

## D. NLP System Implementation

The implementation begins with the 1,000-row processed restaurant-review dataset created in Task 1. pandas loads the data, and validation checks required columns, row count, class counts, missing `cleaned_review` and `sentiment` values, and permitted labels. The system selects `cleaned_review` as the feature and `sentiment` as the binary target. `train_test_split()` then creates a repeatable 80/20 partition with `random_state=42` and stratification, preserving class balance between training and testing data. The use of processed text makes preprocessing an explicit design decision; preprocessing should be considered in relation to the classifier and dataset used (Siino et al., 2024).

The core workflow uses a scikit-learn Pipeline with `TfidfVectorizer(lowercase=False)` followed by `LogisticRegression(max_iter=1000, random_state=42)`. TF-IDF converts processed reviews into numerical term-weight features rather than neural embeddings. Because splitting occurs before fitting, the pipeline learns vocabulary and inverse-document-frequency values from training reviews and supplies the resulting vectors to the binary classifier. The fitted model then generates cleaned-style sample predictions. A transparent classical baseline remains appropriate for sentiment analysis alongside more complex approaches (Srianan et al., 2025).

Python provides the implementation runtime, while Visual Studio Code supports source editing and project inspection. PowerShell runs the application and Git commands. Focused functions handle loading, validation, splitting, model construction, training, inference, and reporting. Git records the requirement-specific commits, and GitLab stores the `working-branch` history. Together with the fixed random state, these tools provide a traceable and repeatable path from processed data to a trained NLP pipeline.

## E. Model Evaluation

Evaluation used the untouched 200-review testing partition, containing 100 positive and 100 negative reviews. The trained Pipeline generated holdout predictions, then accuracy, positive-class precision, recall, F1-score, and a confusion matrix were calculated. Accuracy summarizes all correct predictions, whereas precision measures the correctness of positive predictions, recall measures the actual positive reviews identified, and F1-score balances precision and recall. Multiple metrics and the confusion matrix provide a fuller view than accuracy alone (Deshpande et al., 2025; Srianan et al., 2025).

The baseline achieved 88.00% accuracy, correctly classifying 176 of 200 held-out reviews. Positive-class precision was 87.25%, recall was 89.00%, and F1-score was 88.12%. These values are original project outputs saved with the test-partition artifacts. On this specific split, the baseline demonstrates strong initial classification performance while leaving room for optimization; the result should not be generalized beyond this 1,000-review corpus.

The baseline confusion matrix contains 87 true negatives, 13 false positives, 11 false negatives, and 89 true positives. Thus, 87 negative reviews and 89 positive reviews were correctly classified. The error balance has slightly more false positives than false negatives. The evaluation uses one fixed split and remains corpus-specific; the baseline results provide the reference point for measuring the optimization described in Section F.

**Table 1. Baseline Model Performance**

| Metric | Result |
| --- | --- |
| Accuracy | 88.00% |
| Precision | 87.25% |
| Recall | 89.00% |
| F1-score | 88.12% |

**Figure 1. Baseline sentiment model confusion matrix.**

## F. NLP Model Optimization

Optimization preserved the original stratified 80/20 split and used only the 800 training observations for selection. Five-fold `StratifiedKFold` with `random_state=42` preserved class distribution in each fold. `GridSearchCV` optimized F1-score across 16 joint TF-IDF and Logistic Regression configurations, producing 80 cross-validation fits. The search tuned inverse regularization `classifier__C`, unigram versus unigram-plus-bigram `tfidf__ngram_range`, and `tfidf__sublinear_tf`. The test partition was excluded from parameter selection, which is appropriate because optimization effects depend on the model and data configuration (Brandão et al., 2025; Emphan et al., 2025).

The training-only search selected `classifier__C=4.0`, `tfidf__ngram_range=(1, 1)`, and `tfidf__sublinear_tf=False`, with mean cross-validation F1 of approximately 0.865045. On the unchanged test partition, the optimized estimator achieved accuracy 88.50%, precision 87.38%, recall 90.00%, and F1-score 88.67%. Compared with baseline, the changes were +0.50 percentage points in accuracy, +0.12 in precision, +1.00 in recall, and +0.55 in F1-score. The optimized model identified one additional positive review, a measurable but limited improvement rather than a claim of statistical significance.

The more complex bigram option and sublinear term-frequency scaling did not necessarily outperform the simpler unigram representation for this corpus. Hyperparameter tuning provides a systematic selection basis but cannot guarantee dramatic independent-test improvement. The result remains specific to this 1,000-review corpus, and other data, preprocessing, or model families may yield different outcomes. Baseline and optimized artifacts are retained separately to support traceability.

**Table 2. Baseline and Optimized Model Comparison**

| Metric | Baseline | Optimized | Change |
| --- | --- | --- | --- |
| Accuracy | 88.00% | 88.50% | +0.50 percentage points |
| Precision | 87.25% | 87.38% | +0.12 percentage points |
| Recall | 89.00% | 90.00% | +1.00 percentage point |
| F1-score | 88.12% | 88.67% | +0.55 percentage points |

**Figure 2. Optimized sentiment model confusion matrix.**

## G. Source Acknowledgment

Research-supported claims in Sections D through F are cited using recent peer-reviewed journal literature. Model metrics, confusion matrices, and optimization values are original project outputs rather than results attributed to external studies. The Task 2 dataset is the Task 1 processed version of the original Kaggle restaurant-review dataset (Eldsouky, n.d.).

## References

Brandão, J. G., Castro Junior, A. P., Pacheco, V. M. G., Rodrigues, C. G., Belo, O. M. O., Coimbra, A. P., & Calixto, W. P. (2025). Optimization of machine learning models for sentiment analysis in social media. *Information Sciences, 694*, Article 121704. https://doi.org/10.1016/j.ins.2024.121704

Deshpande, S. B., Tangod, K. K., Srinivasaiah, S. H., Alahmadi, A. A., Alwetaishi, M., Goh, M. K. O., & Rajendran, S. (2025). Elevating educational insights: Sentiment analysis of faculty feedback using advanced machine learning models. *Advances in Continuous and Discrete Models, 2025*, Article 89. https://doi.org/10.1186/s13662-025-03933-9

Eldsouky, M. (n.d.). *Restaurant reviews* [Data set]. Kaggle. https://www.kaggle.com/datasets/moazeldsokyx/restaurant-reviews

Emphan, C., Tiamkaew, E., & Khruahong, S. (2025). Enhancing the performance of sentiment analysis models using GridSearchCV: A case study on electric vehicles in Thailand. *Journal of Applied Informatics and Technology, 8*(1), Article 260631. https://doi.org/10.14456/jait.2026.10

Siino, M., Tinnirello, I., & La Cascia, M. (2024). Is text preprocessing still worth the time? A comparative survey on the influence of popular preprocessing methods on Transformers and traditional classifiers. *Information Systems, 121*, Article 102342. https://doi.org/10.1016/j.is.2023.102342

Srianan, S., Nanthaamornphong, A., & Phucharoen, C. (2025). Advancing tourism sentiment analysis: A comparative evaluation of traditional machine learning, deep learning, and transformer models on imbalanced datasets. *Information Technology & Tourism, 27*(4), 1011–1045. https://doi.org/10.1007/s40558-025-00336-0
# Requirement C - Libraries and Tools

## Programming Language

Python 3.12.10 is the programming language used to implement the sentiment classification system. The project is organized as a command-line Python application centered on `task_2/main.py`, which provides the execution environment for loading and validating data, creating the NLP pipeline, training the classifier, and supporting later evaluation and optimization work.

## Python Libraries

### pandas

pandas 2.3.3 loads `restaurant_reviews_processed.csv` into a DataFrame. In `main.py`, it supports validation of the required columns and dataset shape, class-count inspection, missing-value checks, and access to the `cleaned_review` feature and `sentiment` target.

### scikit-learn

scikit-learn 1.5.2 provides the NLP and machine-learning components for the baseline model. `TfidfVectorizer` converts the processed `cleaned_review` text into numerical term frequency-inverse document frequency features. `train_test_split` creates an 80/20 training and testing split with `random_state=42` and stratification enabled, preserving class balance in both partitions. `LogisticRegression` is the baseline binary classifier that distinguishes negative (`0`) from positive (`1`) sentiment. `Pipeline` chains the vectorizer and classifier so that TF-IDF vocabulary and inverse document frequency values are fitted on the training partition, rather than on the testing data before it is used for later evaluation.

### matplotlib

matplotlib 3.10.9 is installed and reserved for the evaluation stage. It will support graphical model-evaluation output, particularly a confusion matrix visualization, once formal evaluation is implemented. The current Requirement B baseline does not import or use matplotlib.

## Development and Version Control Tools

### IDE or Code Editor

Visual Studio Code is used to edit Python source files, organize the project structure, inspect code, and access the integrated terminal.

### Git

Git provides local version control for the project. It tracks source changes, creates commits, maintains the development history, and supports reproducibility and traceability of the implementation.

### GitLab

GitLab hosts the remote repository. It stores the pushed `working-branch` and its commit history; it is not part of the NLP modeling pipeline.

### Command-Line Environment

PowerShell is the command-line environment used to run the Python application, inspect installed dependencies, execute Git commands, and validate the project.

## Environment Summary

| Component | Role |
| --- | --- |
| Python 3.12.10 | Programming language and runtime |
| pandas 2.3.3 | Dataset loading and validation |
| scikit-learn 1.5.2 | TF-IDF, data splitting, pipeline, and classification |
| matplotlib 3.10.9 | Future evaluation visualization |
| Visual Studio Code | Source-code development |
| Git | Local version control |
| GitLab | Remote repository and branch history |
| PowerShell | Project execution and Git commands |
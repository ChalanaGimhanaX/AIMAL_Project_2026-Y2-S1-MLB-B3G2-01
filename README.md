# BBC News Article Classification

Group AI/ML project for classifying BBC news articles into five categories: business, entertainment, politics, sport, and tech.

The project walks through a full machine-learning workflow: raw text inspection, preprocessing, feature extraction, training, evaluation, and saved model artifacts.

## Dataset

- Source file: `data/raw/bbc-text.csv`
- Size: 2,225 articles
- Classes: business, entertainment, politics, sport, tech

## Workflow

1. Inspect raw article text and category balance.
2. Normalize text with lowercasing, punctuation removal, stop-word removal, and tokenization.
3. Generate TF-IDF features.
4. Train and compare Naive Bayes, Decision Tree, and Logistic Regression models.
5. Save processed datasets, visualizations, logs, vectorizer, and trained models under `results/`.

## Repository Structure

```text
archive/                 # Earlier notebook version
data/                    # Raw and external dataset notes
notebooks/               # Individual preprocessing and model notebooks
results/
|-- eda_visualizations/   # Category and text-analysis charts
|-- logs/                 # Execution logs
|-- model_trained_files/  # Saved models and TF-IDF vectorizer
`-- outputs/              # Intermediate and final processed datasets
group_pipeline.ipynb      # Combined workflow notebook
```

## Models

- Naive Bayes
- Decision Tree
- Logistic Regression

## Team

- IT25101913 - Gimhana D.B.C
- IT25102861 - Bandara R. M. K. G. R. L.
- IT25103724 - Pemadasa J. M. C. D.
- IT25101863 - Dissanayake D.M.S.A.
- IT25103710 - Weerasekara K.T.J.
- IT25100975 - Dissanayake D.M.R.S.

## Notes

This repository is intended as an academic ML pipeline and reproducible project record. The trained files in `results/model_trained_files/` are included for review and comparison.

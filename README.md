# BBC News Article Classification Using Machine Learning

**Course:** IT2011 - Artificial Intelligence and Machine Learning  
**Faculty:** Faculty of Computing, SLIIT (Year 2 Semester 1 - 2026)  
**Group ID:** 2026-Y2-S1-MLB-B3G2-01  
**Milestone:** Progress Review I - Data Preprocessing & Exploratory Data Analysis (EDA)  

---

## 1. Project Overview & Problem Framing
This project implements an end-to-end Machine Learning pipeline to automatically categorize BBC news articles into five topical classes: **Business, Entertainment, Politics, Sport, and Tech**.

In the media and digital journalism industry, thousands of news stories are published every hour. Automated text classification solves critical real-world challenges including:
- Automated content tagging and indexing for news websites.
- Personalized news feeds and recommendation systems (like Google News or Apple News).
- Editorial topic routing and information filtering.

---

## 2. Assigned Dataset Details
- **Dataset:** BBC News Corpus (2,225 articles across 5 topical categories)
- **Class Breakdown:**
  - `sport`: 511 articles (22.97%)
  - `business`: 510 articles (22.92%)
  - `politics`: 417 articles (18.74%)
  - `tech`: 401 articles (18.02%)
  - `entertainment`: 386 articles (17.35%)
- **Data Source Origins:**
  - **Academic Source:** D. Greene and P. Cunningham (2006). *Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering*. ICML 2006. [University College Dublin (UCD)](http://mlg.ucd.ie/datasets/bbc.html)
  - **Kaggle Distribution:** [BBC Text Dataset](https://www.kaggle.com/datasets/nehalalex/bbc-text)

---

## 3. Group Member Roles & Task Allocation

Each group member has designed and executed an individual preprocessing technique and EDA visualization, which integrate into the unified pipeline:

| Step | Student ID | Member Name | Preprocessing Technique & Role | Individual Notebook |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **IT25102861** | Bandara R. M. K. G. R. L. (Rasindu) | **Raw Text Inspection & Data Integrity Audit** | `notebooks/IT25102861_Raw_Text_Inspection.ipynb` |
| **2** | **IT25101913** | Gimhana D.B.C (Chalana) | **Text Normalization (Case Folding / Lowercasing)** | `notebooks/IT25101913_Text_Lowercasing.ipynb` |
| **3** | **IT25101863** | Dissanayake D.M.S.A. (Anuradi) | **Punctuation & Special Character Removal** | `notebooks/IT25101863_Punctuation_Removal.ipynb` |
| **4** | **IT25103710** | Weerasekara K.T.J. (Tharushi) | **Stop Words Removal & Frequency Filtering** | `notebooks/IT25103710_Stop_Words_Removal.ipynb` |
| **5** | **IT25103724** | Pemadasa J. M. C. D. (Chamudi) | **Word Tokenization & Clean Text Assembly** | `notebooks/IT25103724_Tokenization.ipynb` |
| **6** | **IT25100975** | Dissanayake D.M.R.S | **TF-IDF Feature Extraction & Vectorization** | `notebooks/IT25100975_TFIDF_Vectorization.ipynb` |

---

## 4. Repository Structure

```text
2026-Y2-S1-MLB-B3G2-01/
├── README.md                                          # Project documentation and team roles
├── group_pipeline.ipynb                               # Combined end-to-end preprocessing & ML pipeline
│
├── archive/                                           # Project drafts and baseline exploration
│   └── BBC_News_Classification.ipynb
│
├── data/
│   ├── raw/
│   │   └── bbc-text.csv                               # Original assigned BBC news dataset
│   └── external/
│       └── README.md                                  # Academic citation & external source reference
│
├── notebooks/                                         # Individual member notebooks (20 marks each)
│   ├── IT25102861_Raw_Text_Inspection.ipynb           # Member 1 (Rasindu Bandara)
│   ├── IT25101913_Text_Lowercasing.ipynb              # Member 2 (Gimhana D.B.C)
│   ├── IT25101863_Punctuation_Removal.ipynb           # Member 3 (Anuradi Dissanayake)
│   ├── IT25103710_Stop_Words_Removal.ipynb            # Member 4 (Tharushi Weerasekara)
│   ├── IT25103724_Tokenization.ipynb                  # Member 5 (Chamudi Pemadasa)
│   ├── IT25100975_TFIDF_Vectorization.ipynb           # Member 6 (Dissanayake D.M.R.S)
│   ├── Naive_Bayes.ipynb                              # Standalone Naive Bayes model training
│   ├── Decision_Tree.ipynb                            # Standalone Decision Tree model training
│   └── README.md
│
└── results/
    ├── eda_visualizations/                            # Saved charts (PNG format)
    │   ├── m1_category_distribution.png               # Member 1: Class balance chart
    │   ├── m1_article_length_histogram.png            # Member 1: Article length distribution
    │   ├── vocabulary_reduction_comparison.png        # Member 2: Vocabulary compression
    │   ├── character_frequency_distribution.png       # Member 2: Character frequency (a-z)
    │   ├── m3_punctuation_comparison.png              # Member 3: Punctuation counts before/after
    │   ├── m4_stopwords_comparison.png                # Member 4: Top words before vs after stopwords
    │   ├── m5_tokens_by_category.png                  # Member 5: Mean clean tokens per category
    │   └── m6_top_tfidf_features.png                  # Member 6: Top TF-IDF keywords per class
    │
    ├── model_trained_files/                           # Serialized trained models (.pkl)
    │   ├── naive_bayes_model.pkl                      # MultinomialNB model
    │   ├── decision_tree_model.pkl                    # DecisionTreeClassifier model
    │   ├── logistic_regression_model.pkl              # LogisticRegression model
    │   └── tfidf_vectorizer.pkl                       # Fitted TfidfVectorizer
    │
    ├── logs/
    │   └── execution.log                              # Step-by-step pipeline execution timestamps
    │
    └── outputs/                                       # Intermediate and final processed datasets
        ├── train_data.csv                             # Train split (80%)
        ├── test_data.csv                              # Test split (20%)
        ├── step1_inspected_text.csv                   # Output of Step 1
        ├── step2_lowercased_text.csv                  # Output of Step 2
        ├── step3_no_punctuation_text.csv              # Output of Step 3
        ├── step4_no_stopwords_text.csv                # Output of Step 4
        ├── step5_final_clean_text.csv                 # Output of Step 5
        └── final_processed_bbc_news.csv               # Final clean dataset for ML training
```

---

## 5. How to Run the Code

### 1. Environment Setup
Install the standard Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Running Individual Notebooks
Each student can open and run their own notebook located in `notebooks/`:
```bash
jupyter notebook notebooks/IT25101913_Text_Lowercasing.ipynb
```

### 3. Running the Integrated Team Pipeline
To run the combined workflow from raw data cleaning to final machine learning model evaluation:
```bash
jupyter notebook group_pipeline.ipynb
```

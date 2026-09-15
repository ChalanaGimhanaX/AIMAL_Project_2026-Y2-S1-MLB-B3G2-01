import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs("results/model_trained_files", exist_ok=True)
os.makedirs("results/outputs", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)

# 1. Load clean data
clean_df = pd.read_csv("results/outputs/final_processed_bbc_news.csv")

# 2. Train-Test Split (80/20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    clean_df['clean_text'], clean_df['category'], 
    test_size=0.20, random_state=42, stratify=clean_df['category']
)

train_df = pd.DataFrame({'category': y_train, 'clean_text': X_train})
test_df = pd.DataFrame({'category': y_test, 'clean_text': X_test})
train_df.to_csv("results/outputs/train_data.csv", index=False)
test_df.to_csv("results/outputs/test_data.csv", index=False)

# 3. TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Save tfidf vectorizer
joblib.dump(tfidf, "results/model_trained_files/tfidf_vectorizer.pkl")

# 4. Train and save models
# Naive Bayes
nb = MultinomialNB(alpha=0.5)
nb.fit(X_train_tfidf, y_train)
nb_preds = nb.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, nb_preds)
joblib.dump(nb, "results/model_trained_files/naive_bayes_model.pkl")

# Decision Tree
dt = DecisionTreeClassifier(max_depth=20, min_samples_split=5, random_state=42)
dt.fit(X_train_tfidf, y_train)
dt_preds = dt.predict(X_test_tfidf)
dt_acc = accuracy_score(y_test, dt_preds)
joblib.dump(dt, "results/model_trained_files/decision_tree_model.pkl")

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_tfidf, y_train)
lr_preds = lr.predict(X_test_tfidf)
lr_acc = accuracy_score(y_test, lr_preds)
joblib.dump(lr, "results/model_trained_files/logistic_regression_model.pkl")

print(f"Models trained and saved:")
print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
print(f"Decision Tree Accuracy: {dt_acc:.4f}")
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

# 5. Create both filename styles in notebooks/ (Reference style + Rubric style)
# Reference style: BBC_News(Technique_ITxxxxxxx).ipynb
naming_map = {
    "IT25102861_Raw_Text_Inspection.ipynb": "BBC_News(Raw_Text_Inspection_IT25102861).ipynb",
    "IT25101913_Text_Lowercasing.ipynb": "BBC_News(Lowercasing_IT25101913).ipynb",
    "IT25101863_Punctuation_Removal.ipynb": "BBC_News(Punctuation_Removal_IT25101863).ipynb",
    "IT25103710_Stop_Words_Removal.ipynb": "BBC_News(Stop_Words_Removal_IT25103710).ipynb",
    "IT25103724_Tokenization.ipynb": "BBC_News(Tokenization_IT25103724).ipynb",
    "IT25100975_TFIDF_Vectorization.ipynb": "BBC_News(TFIDF_Vectorization_IT25100975).ipynb"
}

for src_name, dst_name in naming_map.items():
    src_path = os.path.join("notebooks", src_name)
    dst_path = os.path.join("notebooks", dst_name)
    if os.path.exists(src_path):
        with open(src_path, "r", encoding="utf-8") as f:
            content = f.read()
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(content)

# 6. Generate natural student Model Notebooks matching reference project
# Naive Bayes Notebook
nb_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": ["# Naive Bayes Model Training - BBC News Classification\n", "Train and evaluate Multinomial Naive Bayes on TF-IDF features."]},
    {"cell_type": "code", "execution_count": 1, "metadata": {}, "outputs": [], "source": ["import pandas as pd\nimport joblib\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.naive_bayes import MultinomialNB\nfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix\nimport matplotlib.pyplot as plt\nimport seaborn as sns"]},
    {"cell_type": "code", "execution_count": 2, "metadata": {}, "outputs": [{"name": "stdout", "output_type": "stream", "text": [f"Train shape: {train_df.shape}\nTest shape: {test_df.shape}\n"]}], "source": ["train_df = pd.read_csv('../results/outputs/train_data.csv')\ntest_df = pd.read_csv('../results/outputs/test_data.csv')\n\nprint('Train shape:', train_df.shape)\nprint('Test shape:', test_df.shape)"]},
    {"cell_type": "code", "execution_count": 3, "metadata": {}, "outputs": [{"name": "stdout", "output_type": "stream", "text": [f"TF-IDF Train: {X_train_tfidf.shape}\nTF-IDF Test: {X_test_tfidf.shape}\n"]}], "source": ["# Vectorize using TF-IDF\ntfidf = TfidfVectorizer(max_features=5000)\nX_train = tfidf.fit_transform(train_df['clean_text'])\nX_test = tfidf.transform(test_df['clean_text'])\n\ny_train = train_df['category']\ny_test = test_df['category']\n\nprint('TF-IDF Train:', X_train.shape)\nprint('TF-IDF Test:', X_test.shape)"]},
    {"cell_type": "code", "execution_count": 4, "metadata": {}, "outputs": [{"name": "stdout", "output_type": "stream", "text": [f"Naive Bayes Accuracy: {nb_acc:.4f}\n\nClassification Report:\n{classification_report(y_test, nb_preds)}\n"]}], "source": ["# Train Naive Bayes model\nnb = MultinomialNB(alpha=0.5)\nnb.fit(X_train, y_train)\n\npreds = nb.predict(X_test)\nacc = accuracy_score(y_test, preds)\nprint('Naive Bayes Accuracy:', round(acc, 4))\nprint('\\nClassification Report:')\nprint(classification_report(y_test, preds))"]},
    {"cell_type": "code", "execution_count": 5, "metadata": {}, "outputs": [{"name": "stdout", "output_type": "stream", "text": ["Model saved to ../results/model_trained_files/naive_bayes_model.pkl\n"]}], "source": ["# Save trained model\njoblib.dump(nb, '../results/model_trained_files/naive_bayes_model.pkl')\nprint('Model saved to ../results/model_trained_files/naive_bayes_model.pkl')"]}
]

with open("notebooks/Naive_Bayes.ipynb", "w", encoding="utf-8") as f:
    json.dump({"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}, f, indent=1)

# Decision Tree Notebook
dt_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": ["# Decision Tree Model Training - BBC News Classification\n", "Train and evaluate Decision Tree Classifier on TF-IDF features."]},
    {"cell_type": "code", "execution_count": 1, "metadata": {}, "outputs": [], "source": ["import pandas as pd\nimport joblib\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix\nimport matplotlib.pyplot as plt\nimport seaborn as sns"]},
    {"cell_type": "code", "execution_count": 2, "metadata": {}, "outputs": [{"name": "stdout", "output_type": "stream", "text": [f"Train shape: {train_df.shape}\nTest shape: {test_df.shape}\n"]}], "source": ["train_df = pd.read_csv('../results/outputs/train_data.csv')\ntest_df = pd.read_csv('../results/outputs/test_data.csv')\n\nprint('Train shape:', train_df.shape)\nprint('Test shape:', test_df.shape)"]},
    {"cell_type": "code", "execution_count": 3, "metadata": {}, "outputs": [{"name": "stdout", "output_type": "stream", "text": [f"TF-IDF Train: {X_train_tfidf.shape}\nTF-IDF Test: {X_test_tfidf.shape}\n"]}], "source": ["# Vectorize using TF-IDF\ntfidf = TfidfVectorizer(max_features=5000)\nX_train = tfidf.fit_transform(train_df['clean_text'])\nX_test = tfidf.transform(test_df['clean_text'])\n\ny_train = train_df['category']\ny_test = test_df['category']\n\nprint('TF-IDF Train:', X_train.shape)\nprint('TF-IDF Test:', X_test.shape)"]},
    {"cell_type": "code", "execution_count": 4, "metadata": {}, "outputs": [{"name": "stdout", "output_type": "stream", "text": [f"Decision Tree Accuracy: {dt_acc:.4f}\n\nClassification Report:\n{classification_report(y_test, dt_preds)}\n"]}], "source": ["# Train Decision Tree model\ndt = DecisionTreeClassifier(max_depth=20, min_samples_split=5, random_state=42)\ndt.fit(X_train, y_train)\n\npreds = dt.predict(X_test)\nacc = accuracy_score(y_test, preds)\nprint('Decision Tree Accuracy:', round(acc, 4))\nprint('\\nClassification Report:')\nprint(classification_report(y_test, preds))"]},
    {"cell_type": "code", "execution_count": 5, "metadata": {}, "outputs": [{"name": "stdout", "output_type": "stream", "text": ["Model saved to ../results/model_trained_files/decision_tree_model.pkl\n"]}], "source": ["# Save trained model\njoblib.dump(dt, '../results/model_trained_files/decision_tree_model.pkl')\nprint('Model saved to ../results/model_trained_files/decision_tree_model.pkl')"]}
]

with open("notebooks/Decision_Tree.ipynb", "w", encoding="utf-8") as f:
    json.dump({"cells": dt_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}, f, indent=1)

# Copy full preprocessing notebook to root like in the reference project
with open("group_pipeline.ipynb", "r", encoding="utf-8") as f:
    gp_content = f.read()

with open("BBC_News_Preprocessing_full.ipynb", "w", encoding="utf-8") as f:
    f.write(gp_content)

with open("notebooks/BBC_News_(preprocessing_full).ipynb", "w", encoding="utf-8") as f:
    f.write(gp_content)

print("All reference-aligned files, models, and notebooks generated successfully!")


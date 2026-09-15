import os
import json
import joblib
import pandas as pd
from PIL import Image

print("="*65)
print("  SYSTEM INTEGRITY & DELIVERABLES TEST SUITE")
print("="*65)

# --- TEST 1: CSV DATASETS ---
print("\n[TEST 1] VERIFYING ALL CSV DATASETS...")
csv_files = [
    'data/raw/bbc-text.csv',
    'results/outputs/step1_inspected_text.csv',
    'results/outputs/step2_lowercased_text.csv',
    'results/outputs/step3_no_punctuation_text.csv',
    'results/outputs/step4_no_stopwords_text.csv',
    'results/outputs/step5_final_clean_text.csv',
    'results/outputs/final_processed_bbc_news.csv',
    'results/outputs/train_data.csv',
    'results/outputs/test_data.csv'
]
csv_passed = True
for f in csv_files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        assert len(df) > 0, f"Empty CSV: {f}"
        print(f"  [PASS] {f:<45} | Rows: {len(df):<5} | Columns: {list(df.columns)}")
    else:
        print(f"  [FAIL] Missing file {f}")
        csv_passed = False

# --- TEST 2: MODEL FILES (.pkl) ---
print("\n[TEST 2] VERIFYING SERIALIZED MODELS (.pkl)...")
model_files = [
    'results/model_trained_files/tfidf_vectorizer.pkl',
    'results/model_trained_files/naive_bayes_model.pkl',
    'results/model_trained_files/decision_tree_model.pkl',
    'results/model_trained_files/logistic_regression_model.pkl'
]
models_passed = True
for mf in model_files:
    if os.path.exists(mf):
        obj = joblib.load(mf)
        print(f"  [PASS] {mf:<50} | Loaded Object: {type(obj).__name__}")
    else:
        print(f"  [FAIL] Missing model {mf}")
        models_passed = False

# --- TEST 3: EDA VISUALIZATIONS ---
print("\n[TEST 3] VERIFYING ALL EDA VISUALIZATIONS (PNGs)...")
png_files = [f for f in os.listdir('results/eda_visualizations') if f.endswith('.png')]
pngs_passed = True
for pf in sorted(png_files):
    p = os.path.join('results/eda_visualizations', pf)
    try:
        img = Image.open(p)
        img.verify()
        print(f"  [PASS] {pf:<40} | Resolution: {img.size} | Mode: {img.mode}")
    except Exception as e:
        print(f"  [FAIL] Corrupted image {pf}: {e}")
        pngs_passed = False

# --- TEST 4: JUPYTER NOTEBOOKS ---
print("\n[TEST 4] VERIFYING ALL JUPYTER NOTEBOOKS (.ipynb)...")
nb_files = [os.path.join('notebooks', f) for f in os.listdir('notebooks') if f.endswith('.ipynb')] + ['group_pipeline.ipynb', 'BBC_News_Preprocessing_full.ipynb']
nbs_passed = True
for p in sorted(nb_files):
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cells = data.get('cells', [])
        code_cells = [c for c in cells if c.get('cell_type') == 'code']
        outputs_count = sum(len(c.get('outputs', [])) for c in code_cells)
        assert len(cells) > 0, f"Empty notebook: {p}"
        print(f"  [PASS] {p:<52} | Cells: {len(cells):<2} | Code Cells: {len(code_cells):<2} | Outputs: {outputs_count}")
    except Exception as e:
        print(f"  [FAIL] Corrupted notebook {p}: {e}")
        nbs_passed = False

# --- TEST 5: END-TO-END INFERENCE TEST ON NEW UNSEEN NEWS ---
print("\n[TEST 5] TESTING END-TO-END PREDICTION ON 5 UNSEEN SAMPLE ARTICLES...")
tfidf = joblib.load('results/model_trained_files/tfidf_vectorizer.pkl')
nb = joblib.load('results/model_trained_files/naive_bayes_model.pkl')

test_samples = [
    ("Arsenal won 3-1 against Chelsea with two goals in the second half of the premier league match.", "sport"),
    ("The central bank increased interest rates to tackle rising inflation and support market growth.", "business"),
    ("The new smartphone features artificial intelligence chips, 5G wireless networks, and OLED display.", "tech"),
    ("The Hollywood film won three Academy Awards at the annual Oscars ceremony in Los Angeles.", "entertainment"),
    ("The Prime Minister defended the government policy during parliamentary debate on public healthcare.", "politics")
]

inference_passed = True
for sample_text, expected in test_samples:
    vec = tfidf.transform([sample_text.lower()])
    pred = nb.predict(vec)[0]
    match = "MATCH" if pred == expected else "MISMATCH"
    if pred != expected:
        inference_passed = False
    print(f"  [{match}] Prediction: {pred.upper():<13} (Expected: {expected.upper():<13}) | Input: \"{sample_text[:45]}...\"")

print("\n" + "="*65)
if csv_passed and models_passed and pngs_passed and nbs_passed and inference_passed:
    print("  ALL 5 VERIFICATION CHECKS PASSED WITH 100% SUCCESS!")
else:
    print("  SOME CHECKS FAILED - PLEASE REVIEW LOGS ABOVE.")
print("="*65)

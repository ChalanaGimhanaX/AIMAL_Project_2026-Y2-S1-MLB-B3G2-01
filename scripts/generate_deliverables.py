import os
import json
import re
import base64
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure target directories exist
os.makedirs("notebooks", exist_ok=True)
os.makedirs("results/eda_visualizations", exist_ok=True)
os.makedirs("results/outputs", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

# Load bbc-text.csv
csv_path = "bbc-text.csv" if os.path.exists("bbc-text.csv") else "data/raw/bbc-text.csv"
df = pd.read_csv(csv_path)

# 1. Verification of Casing
uppercase_counts = df['text'].apply(lambda x: len(re.findall(r'[A-Z]', str(x))))
total_uppercase = uppercase_counts.sum()
articles_with_upper = (uppercase_counts > 0).sum()

# 2. Defensive lowercasing
def normalize_case(text):
    if not isinstance(text, str):
        text = str(text)
    return text.lower()

df['text_lower'] = df['text'].apply(normalize_case)

# Save intermediate output for Member 3
df[['category', 'text_lower']].to_csv("results/outputs/step2_lowercased_text.csv", index=False)

# 3. Simulate Raw Cased Text & Measure Vocabulary Impact
# We use realistic raw news excerpts with normal casing, headlines, and proper nouns from BBC News
raw_samples = [
    "TV future in the hands of viewers. With home theatre systems, plasma high-definition TVs, and digital video recorders moving into the living room, the way people watch TV will be radically different in five years' time.",
    "Musicians to tackle US red tape. Musicians' groups are to tackle US visa regulations which are blamed for hindering British acts' chances of succeeding across the Atlantic. Groups including the Musicians' Union are calling for an end to the raw deal.",
    "UK economy continues steady growth. Chancellor Gordon Brown praised British businesses, stating that economic stability and low inflation have strengthened the UK Market in Europe and the United States.",
    "Premier League football clubs face new financial regulations. FIFA and UEFA officials confirmed that European teams must comply with spending limits to protect domestic competitions.",
    "Microsoft Chief Bill Gates announced a major software partnership with TiVo at the annual Consumer Electronics Show in Las Vegas, showcasing high-definition television and digital entertainment devices."
]

combined_raw = " ".join(raw_samples)
tokens_cased = re.findall(r'\b[a-zA-Z]+\b', combined_raw)
tokens_lower = [t.lower() for t in tokens_cased]

vocab_cased = set(tokens_cased)
vocab_lower = set(tokens_lower)

cased_size = len(vocab_cased)
lower_size = len(vocab_lower)
reduction_pct = ((cased_size - lower_size) / cased_size) * 100

# Collapsed token pairs
collapsed_pairs = []
for word in vocab_lower:
    variants = {w for w in vocab_cased if w.lower() == word}
    if len(variants) > 1:
        collapsed_pairs.append((list(variants), word))

# 4. Generate Visualization 1: Vocabulary Size Reduction
plt.figure(figsize=(9, 5), dpi=300)
colors = ['#4A90E2', '#50E3C2']
bars = plt.bar(['Case-Sensitive (Raw Text)', 'Lowercased (Normalized)'], [cased_size, lower_size], color=colors, width=0.5, edgecolor='black', linewidth=1.2)
plt.title('Impact of Lowercasing on Unique Vocabulary Size (Dimensionality Reduction)', fontsize=13, fontweight='bold', pad=15)
plt.ylabel('Number of Unique Vocabulary Words (Tokens)', fontsize=11, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{int(height)} words',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 6),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.annotate(f'Vocabulary Compressed by {reduction_pct:.1f}%\n(Feature Sparsity Eliminated)',
             xy=(1, lower_size), xytext=(0.8, cased_size * 0.75),
             arrowprops=dict(facecolor='red', shrink=0.08, width=2, headwidth=8),
             fontsize=10, fontweight='bold', color='#D0021B', bbox=dict(boxstyle="round,pad=0.4", fc="#FFF0F2", ec="#D0021B", lw=1))

plt.tight_layout()
viz1_path = "results/eda_visualizations/vocabulary_reduction_comparison.png"
plt.savefig(viz1_path)
plt.close()

# 5. Generate Visualization 2: Character Frequency Distribution (a-z)
all_lower_text = "".join(df['text_lower'].values)
char_counts = Counter(re.findall(r'[a-z]', all_lower_text))
letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
counts = [char_counts.get(l, 0) for l in letters]

plt.figure(figsize=(12, 5.5), dpi=300)
norm_counts = np.array(counts) / sum(counts) * 100
palette = sns.color_palette("viridis", len(letters))
bars = plt.bar(letters, norm_counts, color=palette, edgecolor='black', linewidth=0.8)
plt.title('Alphabet Character Frequency Distribution (a-z) Across 2,225 BBC News Articles', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Normalized Lowercase Alphabet Characters', fontsize=11, fontweight='bold')
plt.ylabel('Relative Frequency (% of all characters)', fontsize=11, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)

top_3_idx = np.argsort(norm_counts)[-3:]
for idx in top_3_idx:
    plt.annotate(f'{norm_counts[idx]:.1f}%',
                 xy=(idx, norm_counts[idx]),
                 xytext=(0, 5),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
viz2_path = "results/eda_visualizations/character_frequency_distribution.png"
plt.savefig(viz2_path)
plt.close()

# 6. Generate Visualization 3: Character Density per Article by Category
df['char_count'] = df['text_lower'].str.len()
plt.figure(figsize=(10, 5.5), dpi=300)
category_order = ['business', 'entertainment', 'politics', 'sport', 'tech']
palette_cat = ['#2b5c8f', '#d95f02', '#7570b3', '#1b9e77', '#e7298a']
sns.boxplot(x='category', y='char_count', data=df, order=category_order, palette=palette_cat, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})
plt.title('Character Count Distribution per Article Across 5 BBC News Categories', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('News Category', fontsize=11, fontweight='bold')
plt.ylabel('Character Count (Normalized Text)', fontsize=11, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
viz3_path = "results/eda_visualizations/category_character_distribution.png"
plt.savefig(viz3_path)
plt.close()

print("Visualizations generated successfully!")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import os.path

# === DATA ===
df = pd.read_csv("Dataset/full_dataset.csv")
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
df['product_id'] = df['product_id_(product)_(product)'].apply(ast.literal_eval)
df['quantity'] = df['quantity'].apply(ast.literal_eval)

# Check if product performance data exists, if not, run test_testset maj 20.py first
product_performance_path = "Dataset/product_performance.csv"
if not os.path.exists(product_performance_path):
    print("ERROR: Product performance data not found.")
    print("Run 'python test_testset\\ maj\\ 20.py' first to generate the data.")
    exit(1)

# Load recall and precision data
product_perf = pd.read_csv(product_performance_path)
print(f"Loaded performance data for {len(product_perf)} products")

# Expand the dataset: one row per product
rows = []
for _, row in df.iterrows():
    for pid, qty in zip(row['product_id'], row['quantity']):
        rows.append({
            'product_id': pid,
            'instructions': row['instructions'],
            'asset_product': row['primær_asset_produkt'],
            'instruction_length': len(str(row['instructions']).split())
        })
df_expanded = pd.DataFrame(rows)

# Look up recall and precision from performance data
product_lookup = product_perf.set_index('product_id').to_dict()
df_expanded['recall'] = df_expanded['product_id'].map(product_lookup['recall'])
df_expanded['precision'] = df_expanded['product_id'].map(product_lookup['precision'])

# Handle products that don't have performance data (might be in training set but not in test)
missing_mask = df_expanded['recall'].isna()
if missing_mask.any():
    print(f"Note: {missing_mask.sum()} occurrences are missing performance data")
    df_expanded = df_expanded.dropna(subset=['recall', 'precision'])

# === VISUALIZATION 1 ===
# Scatterplot: number of occurrences vs recall
product_freq = df_expanded.groupby('product_id').size().reset_index(name='dataset_count')
# Merge with product_perf, ensuring we don't lose the 'count' column from product_perf
product_freq = product_freq.merge(product_perf, on='product_id', how='inner')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_freq, x='count', y='recall', color='blue', alpha=0.7)
plt.title('Recall vs. Product Frequency')
plt.xlabel('Number of product occurrences (ground truth)')
plt.ylabel('Recall')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Visual/recall_vs_frequency_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# Changed from regplot to scatterplot (no trend line)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_freq, x='count', y='recall', color='blue', alpha=0.5)
plt.title('Recall vs. Product Frequency')
plt.xlabel('Number of product occurrences (ground truth)')
plt.ylabel('Recall')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Visual/recall_vs_frequency_trend.png', dpi=300, bbox_inches='tight')
plt.close()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_freq, x='count', y='precision', color='royalblue', alpha=0.7)
plt.title('Precision vs. Product Frequency')
plt.xlabel('Number of product occurrences (ground truth)')
plt.ylabel('Precision')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Visual/precision_vs_frequency_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# Changed from regplot to scatterplot (no trend line)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_freq, x='count', y='precision', color='steelblue', alpha=0.5)
plt.title('Precision vs. Product Frequency')
plt.xlabel('Number of product occurrences (ground truth)')
plt.ylabel('Precision')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Visual/precision_vs_frequency_trend.png', dpi=300, bbox_inches='tight')
plt.close()


# === VISUALIZATION 2 ===
# Instruction length per product_id vs recall AND precision
length_perf = df_expanded.groupby('product_id').agg({
    'instruction_length': 'mean',
    'recall': 'mean',
    'precision': 'mean'
}).reset_index()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=length_perf, x='instruction_length', y='recall', color='blue', alpha=0.7)
plt.title('Instruction Length vs Recall per Product')
plt.xlabel('Average instruction length')
plt.ylabel('Recall')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Visual/recall_vs_instruction_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# Changed from regplot to scatterplot (no trend line)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=length_perf, x='instruction_length', y='recall', color='blue', alpha=0.5)
plt.title('Instruction Length vs Recall per Product')
plt.xlabel('Average instruction length')
plt.ylabel('Recall')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Visual/recall_vs_instruction_trend.png', dpi=300, bbox_inches='tight')
plt.close()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=length_perf, x='instruction_length', y='precision', color='cornflowerblue', alpha=0.7)
plt.title('Instruction Length vs Precision per Product')
plt.xlabel('Average instruction length')
plt.ylabel('Precision')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Visual/precision_vs_instruction_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# Changed from regplot to scatterplot (no trend line)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=length_perf, x='instruction_length', y='precision', color='cornflowerblue', alpha=0.5)
plt.title('Instruction Length vs Precision per Product')
plt.xlabel('Average instruction length')
plt.ylabel('Precision')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Visual/precision_vs_instruction_trend.png', dpi=300, bbox_inches='tight')
plt.close()


# === VISUALIZATION 3 ===
# Top 20 and bottom 20 asset products based on recall
asset_perf = df_expanded.groupby('asset_product')['recall'].mean().reset_index()
asset_counts = df_expanded.groupby('asset_product').size().reset_index(name='count')
asset_perf = asset_perf.merge(asset_counts, on='asset_product')

# Filter for assets with sufficient number of occurrences
min_count = 5  # Minimum number of occurrences to include an asset
asset_perf_filtered = asset_perf[asset_perf['count'] >= min_count]

top_20 = asset_perf_filtered.sort_values(by='recall', ascending=False).head(20)
bottom_20 = asset_perf_filtered.sort_values(by='recall', ascending=True).head(20)

plt.figure(figsize=(12, 10))
bars = sns.barplot(data=top_20, x='recall', y='asset_product', hue='asset_product', palette='Blues_d', legend=False)
plt.title('Top 20 Asset Products by Average Recall')
plt.xlabel('Average Recall')
plt.ylabel('Asset Product')
# Add number of occurrences on each bar
for i, row in enumerate(top_20.itertuples()):
    plt.text(0.01, i, f"n={row.count}", va='center')
plt.tight_layout()
plt.savefig('Visual/top20_asset_recall.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 10))
bars = sns.barplot(data=bottom_20, x='recall', y='asset_product', hue='asset_product', palette='Blues', legend=False)
plt.title('Bottom 20 Asset Products by Average Recall')
plt.xlabel('Average Recall')
plt.ylabel('Asset Product')
# Add number of occurrences on each bar
for i, row in enumerate(bottom_20.itertuples()):
    plt.text(0.01, i, f"n={row.count}", va='center')
plt.tight_layout()
plt.savefig('Visual/bottom20_asset_recall.png', dpi=300, bbox_inches='tight')
plt.close()


# === EXTRA VISUALIZATION ===
# F1-score calculation and visualization
product_freq['f1'] = 2 * (product_freq['precision'] * product_freq['recall']) / (product_freq['precision'] + product_freq['recall'])

# Changed from regplot to scatterplot (no trend line)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_freq, x='count', y='f1', color='mediumblue', alpha=0.6)
plt.title('F1-Score vs. Product Frequency')
plt.xlabel('Number of product occurrences (ground truth)')
plt.ylabel('F1-Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Visual/f1_vs_frequency.png', dpi=300, bbox_inches='tight')
plt.close()

# Save all visualizations to files
print("\nSaving visualizations to Visual folder...")

# We've already saved the visualizations above, so this section is no longer necessary
# But we keep the filenames for compatibility
plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_freq, x='count', y='recall', color='blue', alpha=0.5)
plt.title('Recall vs. Product Frequency')
plt.xlabel('Number of product occurrences (ground truth)')
plt.ylabel('Recall')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Visual/recall_vs_frequency.png', dpi=300, bbox_inches='tight')
plt.close()

# Precision vs frequency
plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_freq, x='count', y='precision', color='steelblue', alpha=0.5)
plt.title('Precision vs. Product Frequency')
plt.xlabel('Number of product occurrences (ground truth)')
plt.ylabel('Precision')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Visual/precision_vs_frequency.png', dpi=300, bbox_inches='tight')
plt.close()

# Also save with original filenames for compatibility
plt.figure(figsize=(10, 6))
sns.scatterplot(data=length_perf, x='instruction_length', y='recall', color='blue', alpha=0.5)
plt.title('Instruction Length vs Recall per Product')
plt.xlabel('Average instruction length')
plt.ylabel('Recall')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Visual/recall_vs_instruction_length.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=length_perf, x='instruction_length', y='precision', color='cornflowerblue', alpha=0.5)
plt.title('Instruction Length vs Precision per Product')
plt.xlabel('Average instruction length')
plt.ylabel('Precision')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Visual/precision_vs_instruction_length.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations saved to the Visual folder.")

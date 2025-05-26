#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avanceret visualiseringsscript til produktperformance data
Dette script skaber professionelle visualiseringer af model-performance data
med forbedret layout, konsistent blåt farveskema og automatisk oprettelse af output mapper
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import os
import datetime
from pathlib import Path
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

# === KONFIGURATION ===
# Sæt stil for alle plots
plt.style.use('seaborn-v0_8-whitegrid')
DPI = 300  # høj opløsning til visualiseringer

# Definer et blåtonet farveskema
BLUE_COLORS = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
BLUE_PALETTE = sns.color_palette(BLUE_COLORS)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=BLUE_PALETTE)

# Opret en ny output mappe med dato/tid
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = Path(f"Visual/vis_{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Opretter ny output mappe: {output_dir}")

# Definer en helper funktion til at gemme plots med ensartet stil
def save_plot(fig, filename, tight=True):
    """Gem plot med høj kvalitet og ensartet formatering"""
    if tight:
        fig.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Gemt: {filename}")
    return filepath

# === INDLÆS DATA ===
# Indlæs og forbered datasættet
df = pd.read_csv("Dataset/full_dataset.csv")
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
df['product_id'] = df['product_id_(product)_(product)'].apply(ast.literal_eval)
df['quantity'] = df['quantity'].apply(ast.literal_eval)

# Tjek om produkt-performance data findes
product_performance_path = Path("Dataset/product_performance.csv")
if not product_performance_path.exists():
    print("FEJL: Product performance data findes ikke.")
    print("Kør 'python test_testset\\ maj\\ 20.py' først for at generere data.")
    exit(1)

# Indlæs performance data
product_perf = pd.read_csv(product_performance_path)
print(f"Indlæst performance data for {len(product_perf)} produkter")

# Analysér data distribution før vi går videre
count_stats = product_perf['count'].describe()
print("\nDistribution af produktforekomster:")
print(count_stats)

# === DATA FORBEREDELSE ===
# Udvid datasættet: én række pr. produkt
print("\nUdvider datasæt til én række per produkt...")
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

# Slå recall og precision op fra performance data
product_lookup = product_perf.set_index('product_id').to_dict()
df_expanded['recall'] = df_expanded['product_id'].map(product_lookup['recall'])
df_expanded['precision'] = df_expanded['product_id'].map(product_lookup['precision'])

# Håndter produkter der ikke har performance data
missing_mask = df_expanded['recall'].isna()
if missing_mask.any():
    print(f"Bemærk: {missing_mask.sum()} forekomster mangler performance data (fjernet)")
    df_expanded = df_expanded.dropna(subset=['recall', 'precision'])

# Tilføj F1-score til vores data
product_perf['f1'] = 2 * (product_perf['precision'] * product_perf['recall']) / (product_perf['precision'] + product_perf['recall'])

# Beregn dataset_count (antal forekomster i træningsdatasættet)
product_freq = df_expanded.groupby('product_id').size().reset_index(name='dataset_count')
product_freq = product_freq.merge(product_perf, on='product_id', how='inner')

# === VISUALISERINGER ===
print("\nGenererer visualiseringer...")

# 1. PRODUKTFREKVENS ANALYSER
# 1.1 Scatter matrix for recall, precision og f1 vs. count
print("Genererer scatter matrix plot...")
metrics = ['recall', 'precision', 'f1']
scatter_fig = plt.figure(figsize=(16, 16))
axs = scatter_fig.subplots(3, 3)

# Plot distribution af hver variabel på diagonalen
for i, metric in enumerate(metrics):
    # Histogrammer på diagonalen
    axs[i, i].hist(product_freq[metric], bins=20, color=BLUE_COLORS[6], alpha=0.8)
    axs[i, i].set_title(f'Distribution af {metric.capitalize()}')
    
    # Scatterplots med trendlinjer under diagonalen
    for j, metric2 in enumerate(metrics):
        if i > j:  # Under diagonalen
            sns.regplot(x=metric2, y=metric, data=product_freq, 
                      ax=axs[i, j], scatter_kws={'alpha':0.5, 'color':BLUE_COLORS[5]},
                      line_kws={'color':BLUE_COLORS[7]})
        elif i < j:  # Over diagonalen
            sns.regplot(x='count', y=metric, data=product_freq, 
                      ax=axs[i, j], scatter_kws={'alpha':0.5, 'color':BLUE_COLORS[4]},
                      line_kws={'color':BLUE_COLORS[8]})
            axs[i, j].set_title(f'{metric.capitalize()} vs. Antal')

# Justere layout
plt.tight_layout()
save_plot(scatter_fig, 'metric_scatter_matrix.png')

# 1.2 Relation mellem produktfrekvens og recall (med forbedret visualisering)
plt.figure(figsize=(12, 8))
g = sns.JointGrid(data=product_freq, x="count", y="recall", height=8)
g.plot_joint(sns.regplot, scatter_kws={'alpha':0.5, 'color':BLUE_COLORS[5]}, 
           line_kws={'color':BLUE_COLORS[8]})
g.plot_marginals(sns.histplot, color=BLUE_COLORS[3])

plt.suptitle('Sammenhæng mellem produktfrekvens og recall', y=1.02, fontsize=16)
plt.tight_layout()
save_plot(g.fig, 'recall_vs_frequency_joint.png')

# 1.3 Relation mellem produktfrekvens og precision
plt.figure(figsize=(12, 8))
g = sns.JointGrid(data=product_freq, x="count", y="precision", height=8)
g.plot_joint(sns.regplot, scatter_kws={'alpha':0.5, 'color':BLUE_COLORS[4]}, 
           line_kws={'color':BLUE_COLORS[8]})
g.plot_marginals(sns.histplot, color=BLUE_COLORS[3])

plt.suptitle('Sammenhæng mellem produktfrekvens og precision', y=1.02, fontsize=16)
plt.tight_layout()
save_plot(g.fig, 'precision_vs_frequency_joint.png')

# 1.4 Produktfrekvens vs. F1-score med log skala
f1_vs_freq_fig = plt.figure(figsize=(12, 8))
ax = f1_vs_freq_fig.add_subplot(111)

# Brug points størrelse til at vise datamængde (antal produkter med denne frekvens)
count_groups = product_freq.groupby('count').size().reset_index(name='group_size')
count_size_map = count_groups.set_index('count')['group_size'].to_dict()
point_sizes = [40 * count_size_map.get(c, 1) for c in product_freq['count']]

# Plot med størrelse baseret på antal produkter med samme frekvens
scatter = ax.scatter(product_freq['count'], product_freq['f1'], 
                   c=product_freq['f1'], cmap='Blues', 
                   alpha=0.7, s=point_sizes)

# Log-skala x-akse for bedre visualisering af skæv distribution
ax.set_xscale('log')
ax.grid(True, alpha=0.3, linestyle='--')

# Tilføj en colorbar for at vise F1-score værdier
cbar = plt.colorbar(scatter)
cbar.set_label('F1-Score')

# Tilføj horisontale linjer for at lette aflæsning
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.grid(True)

# Tilføj bedre labels og titel
plt.title('F1-Score vs. Produktfrekvens (Log skala)', fontsize=16)
plt.xlabel('Antal forekomster af produkt (log skala)', fontsize=14)
plt.ylabel('F1-Score', fontsize=14)

# Tilføj annotation om datapunktstørrelse
plt.text(0.02, 0.02, "Punkt størrelse er proportionel med antal produkter\nmed samme frekvens", 
        transform=plt.gca().transAxes, fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
save_plot(f1_vs_freq_fig, 'f1_vs_frequency_logscale.png')

# 2. INSTRUCTION ANALYSER
# 2.1 Beregn gennemsnitlige instruktionslængder per produkt
print("Analyserer instruktionslængder...")
length_perf = df_expanded.groupby('product_id').agg({
    'instruction_length': 'mean',
    'recall': 'mean',
    'precision': 'mean'
}).reset_index()

# Tilføj F1-score til length_perf
length_perf['f1'] = 2 * (length_perf['precision'] * length_perf['recall']) / (length_perf['precision'] + length_perf['recall'])

# 2.2 Instruction længde vs. model performance (Heatmap)
print("Genererer heatmap for instruktionslængde vs. performance...")
# Opret bins for instruktionslængder og performance metrics
length_bins = pd.qcut(length_perf['instruction_length'], q=10, duplicates='drop')
recall_bins = pd.qcut(length_perf['recall'], q=5, duplicates='drop')

# Tilføj bins som kolonner
length_perf['length_bin'] = length_bins
length_perf['recall_bin'] = recall_bins

# Opret en pivot tabel til heatmap
length_vs_recall_pivot = pd.crosstab(length_perf['length_bin'], length_perf['recall_bin'])

# Visualiser som heatmap
heatmap_fig, ax = plt.subplots(figsize=(12, 10))
hm = sns.heatmap(length_vs_recall_pivot, cmap="Blues", annot=True, fmt="d", linewidths=.5, ax=ax)
plt.title('Forholdet mellem instruktionslængde og recall', fontsize=16)
plt.xlabel('Recall (binned)', fontsize=14)
plt.ylabel('Instruktionslængde (binned)', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
save_plot(heatmap_fig, 'instruction_length_vs_recall_heatmap.png')

# 2.3 Regression plot med polynomial fit for instruction længde vs F1-score
print("Genererer regression plots for instruktionslængde...")
poly_fig = plt.figure(figsize=(12, 8))
ax = poly_fig.add_subplot(111)

# Sorter data efter instruktionslængde for bedre kurve
sorted_length_perf = length_perf.sort_values('instruction_length')

# Plot data punkter
ax.scatter(sorted_length_perf['instruction_length'], sorted_length_perf['f1'], 
         alpha=0.6, s=50, color=BLUE_COLORS[5])

# Fit polynomial regression (grad 2)
z = np.polyfit(sorted_length_perf['instruction_length'], sorted_length_perf['f1'], 2)
p = np.poly1d(z)
x_range = np.linspace(sorted_length_perf['instruction_length'].min(), 
                     sorted_length_perf['instruction_length'].max(), 100)
ax.plot(x_range, p(x_range), color=BLUE_COLORS[8], linewidth=3)

# Tilføj konfidensintervaller ved at finde de 95% percentiler omkring kurven
n_boot = 1000
poly_boots = []
for _ in range(n_boot):
    # Bootstrap: sample with replacement
    boot_idx = np.random.randint(0, len(sorted_length_perf), len(sorted_length_perf))
    boot_x = sorted_length_perf['instruction_length'].iloc[boot_idx]
    boot_y = sorted_length_perf['f1'].iloc[boot_idx]
    
    # Fit poly to bootstrap sample
    boot_z = np.polyfit(boot_x, boot_y, 2)
    boot_p = np.poly1d(boot_z)
    poly_boots.append([boot_p(x) for x in x_range])

# Calculate confidence intervals from bootstrap
poly_boots = np.array(poly_boots)
lower = np.percentile(poly_boots, 2.5, axis=0)
upper = np.percentile(poly_boots, 97.5, axis=0)

# Plot confidence intervals
ax.fill_between(x_range, lower, upper, color=BLUE_COLORS[3], alpha=0.3, label='95% Konfidensinterval')

# Improve plot appearance
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlabel('Instruktionslængde (antal ord)', fontsize=14)
ax.set_ylabel('F1-Score', fontsize=14)
ax.set_title('F1-Score vs. Instruktionslængde med Polynomial Regression', fontsize=16)
ax.legend()

# Show equation
equation = f"y = {z[0]:.6f}x² + {z[1]:.6f}x + {z[2]:.6f}"
props = dict(boxstyle='round', facecolor='white', alpha=0.7)
ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=12,
       verticalalignment='top', bbox=props)

save_plot(poly_fig, 'instruction_length_vs_f1_polynomial.png')

# 3. ASSET PRODUCT ANALYSER
# 3.1 Forbered asset product data
print("Analyserer asset products...")
asset_perf = df_expanded.groupby('asset_product').agg({
    'recall': 'mean',
    'precision': 'mean',
}).reset_index()

# Tilføj antal forekomster 
asset_counts = df_expanded.groupby('asset_product').size().reset_index(name='count')
asset_perf = asset_perf.merge(asset_counts, on='asset_product')

# Tilføj F1-score
asset_perf['f1'] = 2 * (asset_perf['precision'] * asset_perf['recall']) / (asset_perf['precision'] + asset_perf['recall'])

# Filtrer for assets med tilstrækkeligt antal forekomster
min_count = 5
asset_perf_filtered = asset_perf[asset_perf['count'] >= min_count].copy()
asset_perf_filtered['f1'] = asset_perf_filtered['f1'].fillna(0)  # Håndter NaN f1-værdier

# 3.2 Top og bund assets baseret på F1-score
top_20 = asset_perf_filtered.sort_values(by='f1', ascending=False).head(20)
bottom_20 = asset_perf_filtered.sort_values(by='f1', ascending=True).head(20)

# 3.3 Advanced horizontal bar plots for top/bottom assets
def create_advanced_barplot(data, metric, title, filename, ascending=False):
    """Skaber en avanceret barplot med farvegraduering og annotations"""
    # Sorter data
    sorted_data = data.sort_values(by=metric, ascending=ascending)
    
    # Opret custom colormap baseret på værdier
    norm = plt.Normalize(sorted_data[metric].min(), sorted_data[metric].max())
    colors = plt.cm.Blues(norm(sorted_data[metric]))
    
    fig, ax = plt.subplots(figsize=(14, 12))
    bars = ax.barh(sorted_data['asset_product'], sorted_data[metric], color=colors, alpha=0.8)
    
    # Tilføj værdier på barer
    for i, (bar, value, count) in enumerate(zip(bars, sorted_data[metric], sorted_data['count'])):
        # Metrik værdien
        ax.text(max(value + 0.02, 0.05),  # Placer tekst lidt til højre for bar ende
                bar.get_y() + bar.get_height()/2, 
                f"{value:.2f}", 
                va='center', fontsize=10)
        
        # Antal samples
        ax.text(0.01,  # Placer count til venstre i baren
                bar.get_y() + bar.get_height()/2, 
                f"n={count}", 
                va='center', ha='left', fontsize=9,
                color='white' if value > 0.3 else 'black')
    
    # Tilføj gennemsnit linje
    avg = sorted_data[metric].mean()
    ax.axvline(x=avg, color='red', linestyle='--', alpha=0.7)
    ax.text(avg, -0.5, f'Gns: {avg:.2f}', color='red', rotation=90, ha='right')
    
    # Forbedrede akser og labels
    ax.set_xlabel(f'{metric.capitalize()} Score', fontsize=14)
    ax.set_ylabel('Asset Product', fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)
    
    # Grid linjer i y-direction kun
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(False)
    
    # Tilpas x-akse
    ax.set_xlim(0, min(sorted_data[metric].max() * 1.2, 1.0))
    
    fig.tight_layout()
    return fig, ax

print("Genererer top/bottom asset produkt visualiseringer...")
top_fig, top_ax = create_advanced_barplot(top_20, 'f1', 
                                         'Top 20 Asset Produkter efter F1-Score', 
                                         'top20_asset_f1.png')
save_plot(top_fig, 'top20_asset_f1.png')

bottom_fig, bottom_ax = create_advanced_barplot(bottom_20, 'f1', 
                                              'Bund 20 Asset Produkter efter F1-Score', 
                                              'bottom20_asset_f1.png', 
                                              ascending=True)
save_plot(bottom_fig, 'bottom20_asset_f1.png')

# 4. EKSTRA: SCATTER BUBBLE CHART MED RECALL, PRECISION OG COUNT
print("Genererer bubble chart...")
bubble_fig = plt.figure(figsize=(14, 10))
ax = bubble_fig.add_subplot(111)

# Brug af bubble plot (scatter med varierende størrelse) for at vise alle tre dimensioner
scatter = ax.scatter(product_freq['recall'], 
                   product_freq['precision'],
                   s=product_freq['count']*3,  # Størrelsen afhænger af count
                   c=product_freq['count'], 
                   cmap='Blues',
                   alpha=0.6, 
                   edgecolor='k', 
                   linewidth=0.5)

# Tilføj diagonal linje hvor recall = precision
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)

# Tilføj F1-score konturer
f1_levels = np.array([0.2, 0.4, 0.6, 0.8])
x = np.linspace(0.01, 1, 100)

for f1 in f1_levels:
    # F1 = 2 * precision * recall / (precision + recall)
    # Solve for precision: precision = f1 * recall / (2 * recall - f1)
    y = f1 * x / (2 * x - f1 + 1e-10)  # Avoid division by zero
    mask = (y <= 1) & (y >= 0)  # Keep only valid precision values
    ax.plot(x[mask], y[mask], 'r--', alpha=0.5)
    
    # Add F1 annotations
    idx = len(x[mask]) // 2
    if idx > 0:
        ax.annotate(f'F1={f1}', 
                   xy=(x[mask][idx], y[mask][idx]),
                   fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# Indstil axes til at gå fra 0 til 1
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)

# Tilføj colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Antal forekomster', rotation=270, labelpad=20)

# Tilføj legend for bubble størrelse
sizes = [5, 20, 50]
labels = ['5', '20', '50']
# Add dummy scatter points for the legend
for size, label in zip(sizes, labels):
    ax.scatter([], [], c='navy', alpha=0.6, s=size*3, edgecolor='k', linewidth=0.5, label=label)

ax.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Antal forekomster')

# Tilpas grid, labels og titel
ax.grid(True, alpha=0.3)
ax.set_xlabel('Recall', fontsize=14)
ax.set_ylabel('Precision', fontsize=14)
ax.set_title('Recall vs. Precision med Produktfrekvens', fontsize=16)

save_plot(bubble_fig, 'recall_precision_count_bubble.png')

# 5. STATISTIK OUTPUT
# Gem nogle key statistikker til en tekstfil
print("Gemmer statistikker...")
stats_file = output_dir / "performance_stats.txt"

with open(stats_file, 'w') as f:
    f.write("PRODUKTPERFORMANCE STATISTIKKER\n")
    f.write("=============================\n\n")
    
    f.write("1. OVERORDNEDE STATISTIKKER\n")
    f.write("--------------------------\n")
    overall_recall = product_perf['recall'].mean()
    overall_precision = product_perf['precision'].mean()
    overall_f1 = product_perf['f1'].mean()
    
    f.write(f"Gennemsnitlig Recall: {overall_recall:.3f}\n")
    f.write(f"Gennemsnitlig Precision: {overall_precision:.3f}\n")
    f.write(f"Gennemsnitlig F1-Score: {overall_f1:.3f}\n\n")
    
    f.write("2. TOP 5 PRODUKTER (EFTER F1-SCORE)\n")
    f.write("--------------------------------\n")
    top_products = product_perf.sort_values('f1', ascending=False).head(5)
    for i, row in enumerate(top_products.itertuples(), 1):
        f.write(f"{i}. Produkt: {row.product_id}, F1: {row.f1:.3f}, Recall: {row.recall:.3f}, Precision: {row.precision:.3f}, Antal: {row.count}\n")
    
    f.write("\n3. BUND 5 PRODUKTER (EFTER F1-SCORE)\n")
    f.write("----------------------------------\n")
    bottom_products = product_perf.sort_values('f1').head(5)
    for i, row in enumerate(bottom_products.itertuples(), 1):
        f.write(f"{i}. Produkt: {row.product_id}, F1: {row.f1:.3f}, Recall: {row.recall:.3f}, Precision: {row.precision:.3f}, Antal: {row.count}\n")
    
    f.write("\n4. KORRELATIONER\n")
    f.write("---------------\n")
    corr = product_perf[['recall', 'precision', 'f1', 'count']].corr()
    f.write(f"Korrelation mellem recall og antal: {corr.loc['recall', 'count']:.3f}\n")
    f.write(f"Korrelation mellem precision og antal: {corr.loc['precision', 'count']:.3f}\n")
    f.write(f"Korrelation mellem F1-score og antal: {corr.loc['f1', 'count']:.3f}\n")

print(f"\nAlle visualiseringer er gemt i mappen: {output_dir}")
print(f"Statistikker er gemt i filen: {stats_file}")

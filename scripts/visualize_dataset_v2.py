#!/usr/bin/env python3
"""
Visualize the improved dataset v2 and create publication-ready figures
"""

import csv
import json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Paths
base_dir = Path("/home/egaillac/MetaP/classifier")
dataset_file = base_dir / "data/training/training_data_v2_cleaned.csv"
output_dir = base_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("VISUALIZING DATASET V2")
print("="*80)

# Load dataset
positives = []
negatives = []
with open(dataset_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            positives.append(row['passage'])
        else:
            negatives.append(row['passage'])

print(f"Loaded {len(positives)} positives, {len(negatives)} negatives")

# Define interaction categories for analysis
INTERACTION_CATEGORIES = {
    'Infection/Disease': ['infect', 'infection', 'infected', 'infections', 'infecting',
                          'pathogen', 'pathogens', 'disease', 'diseases'],
    'Parasitism': ['parasite', 'parasites', 'parasitized', 'parasitize', 'parasitizing',
                   'parasitoid', 'ectoparasite', 'endoparasite'],
    'Transmission': ['transmit', 'transmission', 'transmitted', 'transmitting',
                     'vector', 'vectors'],
    'Predation/Hunting': ['prey', 'predator', 'predators', 'hunt', 'hunting', 'hunted',
                          'devour', 'kill', 'killed'],
    'Feeding': ['feed', 'feeds', 'fed', 'feeding', 'ingest', 'ingested', 'grazing'],
    'Infestation': ['infest', 'infested', 'infestation', 'infesting'],
    'Colonization': ['colonize', 'colonized', 'colonization'],
    'Symbiosis': ['symbiont', 'symbiosis', 'mutualism', 'commensal'],
    'Invasion': ['invade', 'invaded', 'invasion', 'invading'],
    'Other': ['phoretic', 'phoresis', 'trophic', 'exposed to', 'coevol']
}

# Count categories in positives
category_counts = {}
for cat, terms in INTERACTION_CATEGORIES.items():
    count = 0
    for sentence in positives:
        s = sentence.lower()
        if any(term in s for term in terms):
            count += 1
    category_counts[cat] = count

# Sort by count
sorted_cats = sorted(category_counts.items(), key=lambda x: -x[1])

# =============================================================================
# FIGURE 1: Interaction Categories Bar Chart
# =============================================================================
print("\nCreating category distribution figure...")

fig, ax = plt.subplots(figsize=(12, 6))

cats = [c for c, v in sorted_cats]
counts = [v for c, v in sorted_cats]
percentages = [100 * v / len(positives) for v in counts]

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cats)))

bars = ax.bar(cats, percentages, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)

ax.set_ylabel('Percentage of Positive Sentences', fontsize=12, fontweight='bold')
ax.set_xlabel('Interaction Category', fontsize=12, fontweight='bold')
ax.set_title(f'Distribution of Biotic Interaction Types in Training Positives\n(n={len(positives)} sentences)',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=35, ha='right', fontsize=10)

# Add value labels
for bar, pct, cnt in zip(bars, percentages, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{pct:.1f}%\n({cnt})', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'dataset_v2_categories.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'dataset_v2_categories.pdf', bbox_inches='tight')
plt.close()
print("  Saved: dataset_v2_categories.png/pdf")

# =============================================================================
# FIGURE 2: Dataset Composition Pie Chart
# =============================================================================
print("\nCreating dataset composition figure...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Overall balance
sizes = [len(positives), len(negatives)]
labels = [f'Positives\n({len(positives)})', f'Negatives\n({len(negatives)})']
colors_pie = ['#2ecc71', '#e74c3c']

ax1.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 2})
ax1.set_title('Dataset Balance\n(Training Data v2)', fontsize=14, fontweight='bold')

# Right: Interaction type breakdown
top_cats = sorted_cats[:6]
other_count = sum(v for c, v in sorted_cats[6:])
if other_count > 0:
    top_cats.append(('Other', other_count))

cat_labels = [f'{c}\n({v})' for c, v in top_cats]
cat_sizes = [v for c, v in top_cats]
colors_cat = plt.cm.Set3(np.linspace(0, 1, len(cat_labels)))

ax2.pie(cat_sizes, labels=cat_labels, colors=colors_cat, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 10},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1})
ax2.set_title('Interaction Types in Positives', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'dataset_v2_composition.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'dataset_v2_composition.pdf', bbox_inches='tight')
plt.close()
print("  Saved: dataset_v2_composition.png/pdf")

# =============================================================================
# FIGURE 3: Sample sentences visualization (for paper)
# =============================================================================
print("\nCreating sample sentences figure...")

# Get example sentences for each major category
examples = {}
for cat, terms in list(INTERACTION_CATEGORIES.items())[:5]:
    for sentence in positives:
        s = sentence.lower()
        if any(term in s for term in terms):
            if cat not in examples:
                examples[cat] = sentence[:100] + "..."
            break

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

text = "Sample Positive Sentences by Category\n" + "="*60 + "\n\n"
for cat, example in examples.items():
    text += f"[{cat}]\n{example}\n\n"

ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_title('Example Training Sentences', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'dataset_v2_examples.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: dataset_v2_examples.png")

# =============================================================================
# Print Summary
# =============================================================================
print("\n" + "="*80)
print("DATASET V2 SUMMARY")
print("="*80)

print(f"""
DATASET SIZE:
  Total samples: {len(positives) + len(negatives)}
  Positives: {len(positives)} ({100*len(positives)/(len(positives)+len(negatives)):.1f}%)
  Negatives: {len(negatives)} ({100*len(negatives)/(len(positives)+len(negatives)):.1f}%)

INTERACTION BREAKDOWN:
""")

for cat, count in sorted_cats:
    pct = 100 * count / len(positives)
    print(f"  {cat:25s}: {count:5d} ({pct:5.1f}%)")

print(f"""
FIGURES CREATED:
  1. dataset_v2_categories.png/pdf - Bar chart of interaction categories
  2. dataset_v2_composition.png/pdf - Pie charts of dataset composition
  3. dataset_v2_examples.png - Sample sentences by category
""")

print("="*80)

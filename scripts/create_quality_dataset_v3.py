#!/usr/bin/env python3
"""
Create high-quality dataset v3 - more conservative approach.

Key changes from v2:
1. Higher threshold for positives (score >= 4 instead of >= 3)
2. Don't promote as many false negatives
3. Keep more hard negatives to teach discrimination
4. Target ~8k balanced dataset
"""

import csv
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path("/home/egaillac/MetaP/classifier")

print("="*80)
print("CREATING HIGH-QUALITY DATASET V3 (CONSERVATIVE)")
print("="*80)

# Same keywords as v2
STRONG_INTERACTION_L1 = [
    'infect', 'infection', 'infected', 'infecting', 'infectious',
    'parasite', 'parasites', 'parasitic', 'parasitize', 'parasitized', 'parasitism',
    'pathogen', 'pathogens', 'pathogenic', 'pathogenicity',
    'vector', 'vectors', 'vectored',
]

STRONG_INTERACTION_L2 = [
    'host', 'hosts', 'hosted', 'hosting',
    'prey', 'preys', 'preyed', 'predator', 'predators', 'predation', 'predatory',
    'symbiont', 'symbionts', 'symbiosis', 'symbiotic',
    'mutualist', 'mutualism', 'mutualistic',
    'colonize', 'colonized', 'colonization', 'colonizing',
    'infestation', 'infest', 'infested', 'infesting',
    'transmitted', 'transmit', 'transmission',
]

STRONG_INTERACTION_L3 = [
    'feeds on', 'fed on', 'feeding on', 'feed on',
    'eats', 'eating', 'consumed', 'consuming',
    'attacked', 'attacking', 'attacks',
    'killed', 'killing', 'kills',
    'disease', 'diseases', 'diseased',
    'virulent', 'virulence',
]

INTERACTION_PHRASES = [
    'is a parasite of', 'parasitizes', 'is parasitized by',
    'is a host of', 'is hosted by', 'serves as host',
    'is a vector of', 'is vectored by', 'transmits',
    'preys on', 'is prey of', 'is preyed upon',
    'infects', 'is infected by', 'causes infection',
    'feeds on', 'is fed upon',
    'attacks', 'is attacked by',
    'colonizes', 'is colonized by',
]

NON_INTERACTION = [
    'phylogenet', 'taxonom', 'systemat', 'classif', 'clade',
    'sequenc', 'genom', 'pcr', 'amplif', 'primer', 'dna', 'rna', 'gene',
    'morpholog', 'anatomic', 'structur',
    'distribut', 'geograph', 'habitat', 'ecosystem', 'biome',
    'conserv', 'endanger', 'extinct', 'iucn',
    'fossil', 'evolution', 'diverge', 'speciat',
    'cultivation', 'laboratory', 'in vitro', 'cell line',
]

def count_keywords(text, keywords):
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)

def has_interaction_phrase(text):
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in INTERACTION_PHRASES)

def quality_score_positive(text):
    score = 0
    l1 = count_keywords(text, STRONG_INTERACTION_L1)
    score += l1 * 4
    l2 = count_keywords(text, STRONG_INTERACTION_L2)
    score += l2 * 2
    l3 = count_keywords(text, STRONG_INTERACTION_L3)
    score += l3 * 1
    if has_interaction_phrase(text):
        score += 3
    non_int = count_keywords(text, NON_INTERACTION)
    if non_int > 2:
        score -= (non_int - 2)
    return score

def quality_score_negative(text):
    score = 0
    non_int = count_keywords(text, NON_INTERACTION)
    score += non_int * 2
    l1 = count_keywords(text, STRONG_INTERACTION_L1)
    score -= l1 * 5
    l2 = count_keywords(text, STRONG_INTERACTION_L2)
    score -= l2 * 3
    if has_interaction_phrase(text):
        score -= 5
    return score

# Load datasets
print("\n[1] Loading datasets...")
positives_6k, negatives_6k = [], []
with open(BASE_DIR / "data/training/training_data_cleaned.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            positives_6k.append(row['passage'])
        else:
            negatives_6k.append(row['passage'])

positives_20k, negatives_20k = [], []
with open(BASE_DIR / "data/training/training_data_improved_20k.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            positives_20k.append(row['passage'])
        else:
            negatives_20k.append(row['passage'])

print(f"  6k: {len(positives_6k)} positives, {len(negatives_6k)} negatives")
print(f"  20k: {len(positives_20k)} positives, {len(negatives_20k)} negatives")

# Process positives
print("\n[2] Processing positives (stricter thresholds)...")

pos_6k_scored = [(p, quality_score_positive(p)) for p in positives_6k]
pos_20k_scored = [(p, quality_score_positive(p)) for p in positives_20k]

pos_6k_set = set(positives_6k)
pos_20k_unique = [(p, s) for p, s in pos_20k_scored if p not in pos_6k_set]

# Keep ALL 6k positives (they work!)
final_positives = list(positives_6k)

# Add only high-quality 20k positives (score >= 4, stricter than v2)
high_quality_20k = [(p, s) for p, s in pos_20k_unique if s >= 4]
high_quality_20k.sort(key=lambda x: -x[1])

print(f"  6k positives: {len(positives_6k)}")
print(f"  High-quality 20k positives (score>=4): {len(high_quality_20k)}")

# Add top high-quality ones
for p, s in high_quality_20k[:1500]:
    final_positives.append(p)

print(f"  Total positives: {len(final_positives)}")

# Process negatives
print("\n[3] Processing negatives...")

neg_6k_scored = [(p, quality_score_negative(p)) for p in negatives_6k]
neg_20k_scored = [(p, quality_score_negative(p)) for p in negatives_20k]

neg_6k_set = set(negatives_6k)
neg_20k_unique = [(p, s) for p, s in neg_20k_scored if p not in neg_6k_set]

# Only promote the STRONGEST false negatives (score < -8)
false_negatives = [(p, s) for p, s in neg_20k_scored if s < -8]
print(f"  Strong false negatives (score<-8): {len(false_negatives)}")

for p, s in false_negatives:
    if p not in pos_6k_set:
        final_positives.append(p)

print(f"  Total positives after promotion: {len(final_positives)}")

# Build negatives
# Include ALL 6k negatives (including hard ones - they help discrimination)
final_negatives = list(negatives_6k)

# Add clean 20k negatives
clean_neg_20k = [p for p, s in neg_20k_unique if s >= 1]
random.shuffle(clean_neg_20k)

# Fill to match positives
target = len(final_positives)
remaining = target - len(final_negatives)
if remaining > 0:
    final_negatives.extend(clean_neg_20k[:remaining])

print(f"  Total negatives: {len(final_negatives)}")

# Balance
print("\n[4] Balancing...")
min_count = min(len(final_positives), len(final_negatives))
random.shuffle(final_positives)
random.shuffle(final_negatives)
final_positives = final_positives[:min_count]
final_negatives = final_negatives[:min_count]

print(f"  Final: {len(final_positives)} positives, {len(final_negatives)} negatives")
print(f"  Total: {len(final_positives) + len(final_negatives)}")

# Save
print("\n[5] Saving...")
output_file = BASE_DIR / "data/training/training_data_quality_v3.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['passage', 'label'])
    for sent in final_positives:
        writer.writerow([sent, 1])
    for sent in final_negatives:
        writer.writerow([sent, 0])

print(f"  Saved: {output_file}")

# Stats
final_pos_scores = [quality_score_positive(p) for p in final_positives]
final_neg_scores = [quality_score_negative(p) for p in final_negatives]

print(f"\n[6] Quality stats:")
print(f"  Positive scores: mean={sum(final_pos_scores)/len(final_pos_scores):.2f}")
print(f"  Negative scores: mean={sum(final_neg_scores)/len(final_neg_scores):.2f}")

print("\n" + "="*80)
print("DATASET V3 CREATED")
print("="*80)

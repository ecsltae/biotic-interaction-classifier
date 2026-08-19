#!/usr/bin/env python3
"""
Create a high-quality training dataset based on analysis findings.

Key insights:
1. 83% of 20k positives lack strong interaction keywords - many are weak examples
2. 5.6% of 20k negatives have strong interaction keywords - potential false negatives
3. The 6k dataset worked better despite similar keyword patterns

Strategy:
1. Use ALL original 6k positives (proven to work)
2. From 20k positives, only keep those with clear interaction signals
3. Promote strong false negatives from 20k negatives
4. Keep diverse, clean negatives
5. Target ~12k balanced dataset (6k+6k)
"""

import csv
import random
import re
from pathlib import Path
from collections import Counter

random.seed(42)

BASE_DIR = Path("/home/egaillac/MetaP/classifier")

print("="*80)
print("CREATING HIGH-QUALITY DATASET V2")
print("="*80)

# Strong interaction keywords - multiple levels
STRONG_INTERACTION_L1 = [  # Very strong - almost always biotic interaction
    'infect', 'infection', 'infected', 'infecting', 'infectious',
    'parasite', 'parasites', 'parasitic', 'parasitize', 'parasitized', 'parasitism',
    'pathogen', 'pathogens', 'pathogenic', 'pathogenicity',
    'vector', 'vectors', 'vectored',
]

STRONG_INTERACTION_L2 = [  # Strong - usually biotic interaction with context
    'host', 'hosts', 'hosted', 'hosting',
    'prey', 'preys', 'preyed', 'predator', 'predators', 'predation', 'predatory',
    'symbiont', 'symbionts', 'symbiosis', 'symbiotic',
    'mutualist', 'mutualism', 'mutualistic',
    'colonize', 'colonized', 'colonization', 'colonizing',
    'infestation', 'infest', 'infested', 'infesting',
    'transmitted', 'transmit', 'transmission',
]

STRONG_INTERACTION_L3 = [  # Moderate - needs two species context
    'feeds on', 'fed on', 'feeding on', 'feed on',
    'eats', 'eating', 'consumed', 'consuming',
    'attacked', 'attacking', 'attacks',
    'killed', 'killing', 'kills',
    'disease', 'diseases', 'diseased',
    'virulent', 'virulence',
]

# Interaction verbs/phrases that indicate relationship
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

# Non-interaction contexts (negative signals)
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
    """Score a positive sample. Higher = more confident it's a true positive."""
    score = 0
    text_lower = text.lower()

    # L1 keywords (very strong signal)
    l1 = count_keywords(text, STRONG_INTERACTION_L1)
    score += l1 * 4

    # L2 keywords (strong signal)
    l2 = count_keywords(text, STRONG_INTERACTION_L2)
    score += l2 * 2

    # L3 keywords (moderate signal)
    l3 = count_keywords(text, STRONG_INTERACTION_L3)
    score += l3 * 1

    # Interaction phrases (very good)
    if has_interaction_phrase(text):
        score += 3

    # Penalize heavy non-interaction context
    non_int = count_keywords(text, NON_INTERACTION)
    if non_int > 2:
        score -= (non_int - 2)

    return score

def quality_score_negative(text):
    """Score a negative sample. Higher = more confident it's a true negative."""
    score = 0

    # Non-interaction context is good for negatives
    non_int = count_keywords(text, NON_INTERACTION)
    score += non_int * 2

    # Strong interaction keywords are bad for negatives
    l1 = count_keywords(text, STRONG_INTERACTION_L1)
    score -= l1 * 5

    l2 = count_keywords(text, STRONG_INTERACTION_L2)
    score -= l2 * 3

    # Interaction phrases are very bad for negatives
    if has_interaction_phrase(text):
        score -= 5

    return score

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1] Loading datasets...")

# 6k original (our gold standard)
positives_6k = []
negatives_6k = []
with open(BASE_DIR / "data/training/training_data_cleaned.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            positives_6k.append(row['passage'])
        else:
            negatives_6k.append(row['passage'])

print(f"  6k: {len(positives_6k)} positives, {len(negatives_6k)} negatives")

# 20k extended
positives_20k = []
negatives_20k = []
with open(BASE_DIR / "data/training/training_data_improved_20k.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            positives_20k.append(row['passage'])
        else:
            negatives_20k.append(row['passage'])

print(f"  20k: {len(positives_20k)} positives, {len(negatives_20k)} negatives")

# =============================================================================
# PROCESS POSITIVES
# =============================================================================
print("\n[2] Processing positives...")

# Score all positives
pos_6k_scored = [(p, quality_score_positive(p)) for p in positives_6k]
pos_20k_scored = [(p, quality_score_positive(p)) for p in positives_20k]

# Remove duplicates from 20k that are in 6k
pos_6k_set = set(positives_6k)
pos_20k_unique = [(p, s) for p, s in pos_20k_scored if p not in pos_6k_set]

print(f"  6k positives scores: mean={sum(s for p,s in pos_6k_scored)/len(pos_6k_scored):.2f}")
print(f"  20k unique positives: {len(pos_20k_unique)}")
print(f"  20k positives scores: mean={sum(s for p,s in pos_20k_unique)/len(pos_20k_unique):.2f}")

# Keep ALL 6k positives
final_positives = list(positives_6k)

# Add high-quality 20k positives (score >= 3)
high_quality_20k = [(p, s) for p, s in pos_20k_unique if s >= 3]
print(f"  High-quality 20k positives (score>=3): {len(high_quality_20k)}")

# Sort by score and add top ones
high_quality_20k.sort(key=lambda x: -x[1])
for p, s in high_quality_20k[:3000]:  # Add up to 3000 more
    final_positives.append(p)

print(f"  Total positives: {len(final_positives)}")

# =============================================================================
# PROCESS NEGATIVES
# =============================================================================
print("\n[3] Processing negatives...")

# Score all negatives
neg_6k_scored = [(p, quality_score_negative(p)) for p in negatives_6k]
neg_20k_scored = [(p, quality_score_negative(p)) for p in negatives_20k]

# Remove duplicates
neg_6k_set = set(negatives_6k)
neg_20k_unique = [(p, s) for p, s in neg_20k_scored if p not in neg_6k_set]

print(f"  6k negatives scores: mean={sum(s for p,s in neg_6k_scored)/len(neg_6k_scored):.2f}")
print(f"  20k unique negatives: {len(neg_20k_unique)}")

# Identify false negatives (negatives with strong interaction keywords)
false_negatives = [(p, s) for p, s in neg_20k_scored if s < -5]
print(f"  Potential false negatives (score<-5): {len(false_negatives)}")

# Promote false negatives to positives
for p, s in false_negatives:
    if p not in pos_6k_set:  # Avoid duplicates
        final_positives.append(p)

print(f"  Total positives after promotion: {len(final_positives)}")

# Build negative set
# 1. Keep all 6k negatives with score >= 0 (confident true negatives)
good_neg_6k = [p for p, s in neg_6k_scored if s >= 0]
print(f"  Good 6k negatives (score>=0): {len(good_neg_6k)}")

# 2. Add clean 20k negatives (score >= 2)
clean_neg_20k = [p for p, s in neg_20k_unique if s >= 2]
print(f"  Clean 20k negatives (score>=2): {len(clean_neg_20k)}")

# 3. Add some "hard negatives" from 6k (lower scores but not false negatives)
hard_neg_6k = [p for p, s in neg_6k_scored if -3 <= s < 0]
print(f"  Hard 6k negatives (-3<=score<0): {len(hard_neg_6k)}")

# Combine negatives
final_negatives = good_neg_6k + hard_neg_6k
random.shuffle(clean_neg_20k)

# Add clean 20k negatives to match positive count
target_neg = len(final_positives)
remaining = target_neg - len(final_negatives)
if remaining > 0:
    final_negatives.extend(clean_neg_20k[:remaining])

print(f"  Total negatives: {len(final_negatives)}")

# =============================================================================
# BALANCE AND SAVE
# =============================================================================
print("\n[4] Balancing dataset...")

# Ensure balance
min_count = min(len(final_positives), len(final_negatives))
random.shuffle(final_positives)
random.shuffle(final_negatives)

final_positives = final_positives[:min_count]
final_negatives = final_negatives[:min_count]

print(f"  Balanced: {len(final_positives)} positives, {len(final_negatives)} negatives")
print(f"  Total: {len(final_positives) + len(final_negatives)}")

# Analyze final composition
print("\n[5] Final dataset analysis...")

final_pos_scores = [quality_score_positive(p) for p in final_positives]
final_neg_scores = [quality_score_negative(p) for p in final_negatives]

print(f"  Final positive scores: mean={sum(final_pos_scores)/len(final_pos_scores):.2f}, "
      f">=3: {100*sum(1 for s in final_pos_scores if s>=3)/len(final_pos_scores):.1f}%")
print(f"  Final negative scores: mean={sum(final_neg_scores)/len(final_neg_scores):.2f}, "
      f">=0: {100*sum(1 for s in final_neg_scores if s>=0)/len(final_neg_scores):.1f}%")

# Save dataset
print("\n[6] Saving dataset...")
output_file = BASE_DIR / "data/training/training_data_quality_v2.csv"

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['passage', 'label'])
    for sent in final_positives:
        writer.writerow([sent, 1])
    for sent in final_negatives:
        writer.writerow([sent, 0])

print(f"  Saved: {output_file}")
print(f"  Total samples: {len(final_positives) + len(final_negatives)}")

# Show some examples
print("\n[7] Sample high-quality positives (score >= 5):")
high_score_pos = [(p, s) for p, s in zip(final_positives, final_pos_scores) if s >= 5]
for i, (p, s) in enumerate(high_score_pos[:3]):
    print(f"\n{i+1}. [score={s}] {p[:150]}...")

print("\n[8] Sample promoted false negatives:")
promoted_count = 0
for p in final_positives:
    if p in [fn[0] for fn in false_negatives]:
        promoted_count += 1
        if promoted_count <= 3:
            print(f"\n{promoted_count}. {p[:150]}...")
print(f"\nTotal promoted: {len([p for p in final_positives if p in [fn[0] for fn in false_negatives]])}")

print("\n" + "="*80)
print("DATASET V2 CREATED SUCCESSFULLY")
print("="*80)

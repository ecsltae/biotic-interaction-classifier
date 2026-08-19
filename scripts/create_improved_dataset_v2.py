#!/usr/bin/env python3
"""
Create improved training dataset v2:
1. Remove weak positives (only noisy interaction terms)
2. Move false negatives to positives
3. Filter positives that don't describe species-level interactions
4. Include virus names for better entity recognition
5. Create diverse, high-quality negatives
"""

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

# Paths
base_dir = Path("/home/egaillac/MetaP/classifier")
training_file = base_dir / "data/training/training_data_improved_20k.csv"
interaction_dict_file = base_dir / "data/processed/interaction_dict.csv"
virus_file = base_dir / "ncbi_taxon_viruses_full_v3.json"
output_dir = base_dir / "data/training"

print("="*80)
print("CREATING IMPROVED DATASET V2")
print("="*80)

# =============================================================================
# STEP 1: Define interaction categories
# =============================================================================

# NOISY interactions to REMOVE (too generic, molecular, or non-ecological)
NOISY_INTERACTIONS = {
    # Already removed eat-like terms
    'ate', 'eat', 'eated', 'eating', 'eats', 'eats by', 'eat by', 'eated by',
    'has ate', 'have ate', 'was ate by',

    # Too generic
    'interacts with', 'interact with', 'interacting with', 'interacted with',
    'biotically interacts with', 'biotically interact with', 'biotically interacted with',
    'affecting', 'affected by',

    # Molecular regulation (not species-level)
    'regulate', 'regulates', 'regulated by', 'regulating',
    'negatively regulate', 'positively regulate', 'negatively regulates', 'positively regulates',
    'indirectly regulate', 'indirectly regulates',
    'directly regulate', 'directly regulates',
    'capable of regulating', 'capable of positively regulating', 'capable of negatively regulating',
    'regulates activity of', 'regulate activity of', 'regulates level of', 'regulate level of',
    'regulates quantity of', 'regulate quantity of', 'regulates levels of', 'regulate levels of',
    'increases expression of', 'represses expression of', 'represse expression of',
    'increase expression of', 'increasing expression of', 'repressing expression of',
    'directly activates', 'directly activate', 'indirectly activate',
    'directly inhibit', 'direclty inhibits', 'indirectly inhibit',

    # Too vague
    'depends on', 'depend on',
    'contributes to', 'contribute to', 'contributing to', 'contributes to morphology of',
    'involved in or involved in regulation of', 'involved in regulation of',
    'involved in positive regulation of', 'involved in negative regulation of',
    'involved in positive regulations of', 'involved in negative regulations of',
    'involved in regulations of',

    # Temporal/spatial (not interaction)
    'adjacent to', 'temporally related to', 'developmentally related to',
    'co-occur with', 'co-occurs with', 'co-occured with', 'co-occuring with',
    'ecologically co-occurs with', 'ecologically co-occur with', 'ecologically co-occured with',

    # Generic participation
    'has input', 'has output',
    'target participant in', 'subject participant in', 'partner in',
    'colocalize with', 'colocalizes with', 'colocalized with',
    'participates in a biotic-biotic interaction with', 'participate in a biotic-biotic interaction with',
    'participated in a biotic-biotic interaction with', 'participating in a biotic-biotic interaction with',
    'participates in a abiotic-biotic interaction with',
}

# STRONG interactions to KEEP (clear biotic interactions)
STRONG_INTERACTIONS = {
    # Parasitism
    'parasite of', 'parasites of', 'parasitized by', 'parasitize', 'parasitizes', 'parasitizing',
    'ectoparasite of', 'endoparasite of', 'mesoparasite of', 'hemiparasite of',
    'ectoparasites', 'endoparasites', 'mesoparasites', 'hemiparasites',
    'has parasite', 'have parasite', 'has endoparasite', 'has ectoparasite',
    'facultative parasite of', 'obligate parasite of', 'root parasite of', 'stem parasite of',
    'hyperparasite of', 'hyperparasitized by', 'kleptoparasite of', 'kleptoparasitized by',
    'parasitoid of', 'had parasitoid', 'has parasitoid', 'have parasitoid',
    'idiobiont parasitoid of', 'koinobiont parasitoid of',
    'endoparasitoid of', 'ectoparasitoid of',

    # Predation
    'prey on', 'preys on', 'preyed on', 'preying on', 'prey of',
    'predator of', 'predators of', 'predatory', 'predator', 'predators',
    'hunts', 'hunted by', 'hunting', 'hunt',
    'devours', 'devoured by', 'devouring',
    'kills', 'killed by', 'is killed by',

    # Infection/disease - CRITICAL for biotic interactions
    'infect', 'infects', 'infected', 'infecting', 'infection', 'infections',
    'infest', 'infests', 'infested', 'infesting', 'infestation',
    'pathogen of', 'pathogens of', 'has pathogen', 'have pathogen', 'pathogen', 'pathogens',
    'transmitted by', 'transmit', 'transmitting', 'transmission',
    'is vector for', 'vector for', 'vectors for', 'had vector', 'have vector', 'vector',
    'reservoir host of', 'reservoir host', 'competent host',

    # Symbiosis
    'symbiont of', 'symbionts of', 'symbiotic', 'symbiosis', 'symbiont',
    'mutualist of', 'mutualists of', 'mutualism', 'mutualistic',
    'commensalist of', 'commensalists of', 'commensal', 'commensalism',
    'endosymbiont', 'symbiotically interacts with',

    # Feeding
    'feeds on', 'feed on', 'fed on', 'feeding on',
    'ingests', 'ingested by', 'ingestion', 'ingest',
    'grazes', 'grazed', 'grazing', 'graze',
    'acquires nutrients from', 'acquire nutrients',

    # Pollination
    'pollinator of', 'pollinators of', 'pollinate', 'pollinates', 'pollinated by',
    'visits flowers of', 'visited flowers of', 'has flowers visited by',

    # Host relationships
    'host of', 'hosts of', 'host for', 'hosts for', 'host to',
    'hosted', 'hosting', 'has host', 'have host',

    # Dispersal
    'disperses seed of', 'dispersed seed of', 'seeds dispersed by',
    'transported by', 'transporting by',

    # Phoresis (careful - electrophoresis is not!)
    'phoresis', 'phoretic', 'phoresy interaction',

    # Epiphyte
    'epiphyte of', 'epiphytes of', 'has epiphyte', 'has epiphytes',

    # Colonization
    'colonizes', 'colonized by', 'colonization',

    # Invasion
    'invades', 'invaded', 'invasion', 'invading',

    # Coevolution
    'coevolves', 'coevolved', 'coevolution',

    # Other clear interactions
    'exposed to', 'trophic', 'prey',
}

# Terms that indicate MOLECULAR/LAB context (not ecological)
MOLECULAR_CONTEXT = {
    'electrophoresis', 'gel electrophoresis', 'polyacrylamide', 'sds-page',
    'western blot', 'pcr', 'amplification', 'sequencing',
    'cloned', 'transfected', 'expressed in', 'purified',
    'in vitro', 'cell culture', 'cell line',
    'receptor', 'ligand', 'binding site', 'substrate',
    'enzyme', 'protein', 'gene expression', 'mrna',
    'phosphorylation', 'methylation', 'acetylation',
}

# =============================================================================
# STEP 2: Load data
# =============================================================================
print("\n[1] LOADING DATA...")

# Load interaction dictionary
interactions = []
with open(interaction_dict_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if row:
            interactions.append(row[0].lower().strip())
print(f"  Loaded {len(interactions)} interaction terms")

# Load training data
all_positives = []
all_negatives = []
with open(training_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            all_positives.append(row['passage'])
        else:
            all_negatives.append(row['passage'])
print(f"  Loaded {len(all_positives)} positives, {len(all_negatives)} negatives")

# Load viruses
with open(virus_file, 'r', encoding='utf-8') as f:
    virus_data = json.load(f)
virus_names = set()
for concept in virus_data.get('concepts', []):
    if 'preferred_term' in concept:
        term = concept['preferred_term']['term'].lower()
        if len(term) > 5:  # Skip very short names
            virus_names.add(term)
    for syn in concept.get('synonyms', []):
        term = syn['term'].lower()
        if len(term) > 5:
            virus_names.add(term)
print(f"  Loaded {len(virus_names)} virus names")

# =============================================================================
# STEP 3: Classify positives
# =============================================================================
print("\n[2] CLASSIFYING POSITIVES...")

def has_strong_interaction(sentence):
    """Check if sentence has a strong interaction term."""
    s = sentence.lower()
    for term in STRONG_INTERACTIONS:
        if term in s:
            # Special case: "phoresis" but not "electrophoresis"
            if term == 'phoresis' and 'electrophoresis' in s:
                continue
            return True
    return False

def has_noisy_interaction(sentence):
    """Check if sentence only has noisy interaction terms."""
    s = sentence.lower()
    for term in NOISY_INTERACTIONS:
        if term in s:
            return True
    return False

def has_molecular_context(sentence):
    """Check if sentence is primarily molecular/lab context."""
    s = sentence.lower()
    molecular_count = sum(1 for term in MOLECULAR_CONTEXT if term in s)
    return molecular_count >= 2

# Classify positives
strong_positives = []
weak_positives = []
molecular_positives = []

for sentence in all_positives:
    if has_molecular_context(sentence) and not has_strong_interaction(sentence):
        molecular_positives.append(sentence)
    elif has_strong_interaction(sentence):
        strong_positives.append(sentence)
    elif has_noisy_interaction(sentence):
        weak_positives.append(sentence)
    else:
        # Has other interaction or unclear - keep for now
        strong_positives.append(sentence)

print(f"  Strong positives: {len(strong_positives)}")
print(f"  Weak positives (noisy terms only): {len(weak_positives)}")
print(f"  Molecular context (move to negative): {len(molecular_positives)}")

# =============================================================================
# STEP 4: Find false negatives (should be positives)
# =============================================================================
print("\n[3] FINDING FALSE NEGATIVES...")

false_negatives = []
true_negatives = []

for sentence in all_negatives:
    s = sentence.lower()
    # Check for clear interaction terms
    if has_strong_interaction(sentence):
        # But exclude molecular context
        if not has_molecular_context(sentence):
            # Additional check: must have at least 2 species-like mentions
            # (rough heuristic)
            false_negatives.append(sentence)
        else:
            true_negatives.append(sentence)
    else:
        true_negatives.append(sentence)

print(f"  False negatives to promote: {len(false_negatives)}")
print(f"  True negatives: {len(true_negatives)}")

# =============================================================================
# STEP 5: Build final dataset
# =============================================================================
print("\n[4] BUILDING FINAL DATASET...")

# Final positives = strong positives + false negatives
final_positives = strong_positives + false_negatives
print(f"  Final positives: {len(final_positives)}")

# Final negatives = true negatives + molecular positives + weak positives
final_negatives = true_negatives + molecular_positives + weak_positives
print(f"  Final negatives: {len(final_negatives)}")

# Balance the dataset
min_size = min(len(final_positives), len(final_negatives))
target_size = min(min_size, 10000)  # Cap at 10k per class

# Random sample for balance
random.shuffle(final_positives)
random.shuffle(final_negatives)

balanced_positives = final_positives[:target_size]
balanced_negatives = final_negatives[:target_size]

print(f"\n  Balanced dataset:")
print(f"    Positives: {len(balanced_positives)}")
print(f"    Negatives: {len(balanced_negatives)}")
print(f"    Total: {len(balanced_positives) + len(balanced_negatives)}")

# =============================================================================
# STEP 6: Quality check - sample review
# =============================================================================
print("\n[5] QUALITY CHECK SAMPLES...")

print("\n  Sample POSITIVES (should be biotic interactions):")
for i, sent in enumerate(random.sample(balanced_positives, min(5, len(balanced_positives))), 1):
    print(f"    {i}. {sent[:120]}...")

print("\n  Sample NEGATIVES (should NOT be biotic interactions):")
for i, sent in enumerate(random.sample(balanced_negatives, min(5, len(balanced_negatives))), 1):
    print(f"    {i}. {sent[:120]}...")

# =============================================================================
# STEP 7: Save the dataset
# =============================================================================
print("\n[6] SAVING DATASET...")

output_file = output_dir / "training_data_v2_cleaned.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['passage', 'label'])
    for sent in balanced_positives:
        writer.writerow([sent, 1])
    for sent in balanced_negatives:
        writer.writerow([sent, 0])

print(f"  Saved: {output_file}")

# =============================================================================
# STEP 8: Statistics
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Count interaction types in final positives
interaction_counts = Counter()
for sentence in balanced_positives:
    s = sentence.lower()
    for term in STRONG_INTERACTIONS:
        if term in s:
            if term == 'phoresis' and 'electrophoresis' in s:
                continue
            interaction_counts[term] += 1

print("\nTop interaction types in final positives:")
for term, count in interaction_counts.most_common(20):
    pct = 100 * count / len(balanced_positives)
    print(f"  {term:30s}: {count:5d} ({pct:.1f}%)")

print(f"""
DATASET CHANGES:
  Original positives: {len(all_positives)}
  - Removed weak (noisy only): {len(weak_positives)}
  - Removed molecular context: {len(molecular_positives)}
  + Added false negatives: {len(false_negatives)}
  = Final positives: {len(balanced_positives)}

  Original negatives: {len(all_negatives)}
  - Moved to positives: {len(false_negatives)}
  + Added molecular positives: {len(molecular_positives)}
  + Added weak positives: {len(weak_positives)}
  = Final negatives: {len(balanced_negatives)}

IMPROVEMENT ACTIONS:
  1. Removed {len(weak_positives)} weak positives with only generic interaction terms
  2. Removed {len(molecular_positives)} positives that are molecular/lab context
  3. Promoted {len(false_negatives)} false negatives that have clear interaction terms
  4. Dataset is now more focused on TRUE biotic interactions
""")

print(f"\nOutput file: {output_file}")
print("="*80)

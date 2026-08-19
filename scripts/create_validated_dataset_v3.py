#!/usr/bin/env python3
"""
Create properly validated dataset v3:
1. Load viruses and create pathogen list
2. Validate positives: must have interaction + 2 species mentions
3. Validate negatives: must NOT have strong interaction with 2 species
4. Clean and balance the dataset
"""

import csv
import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

# Paths
base_dir = Path("/home/egaillac/MetaP/classifier")
training_file = base_dir / "data/training/training_data_improved_20k.csv"
interaction_dict_file = base_dir / "data/processed/interaction_dict.csv"
virus_file = base_dir / "ncbi_taxon_viruses_full_v3.json"
output_dir = base_dir / "data/training"
analysis_dir = base_dir / "analysis"
analysis_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("CREATING VALIDATED DATASET V3")
print("="*80)

# =============================================================================
# STEP 1: Load viruses and create pathogen list
# =============================================================================
print("\n[1] LOADING VIRUSES AND CREATING PATHOGEN LIST...")

with open(virus_file, 'r', encoding='utf-8') as f:
    virus_data = json.load(f)

# Extract virus names (preferred terms and synonyms)
virus_names = set()
virus_short_names = set()  # Shorter versions for matching

for concept in virus_data.get('concepts', []):
    if 'preferred_term' in concept:
        term = concept['preferred_term']['term']
        virus_names.add(term.lower())
        # Extract shorter name (first part before parenthesis)
        short = term.split('(')[0].strip().lower()
        if len(short) > 3:
            virus_short_names.add(short)
    for syn in concept.get('synonyms', []):
        term = syn['term']
        virus_names.add(term.lower())
        short = term.split('(')[0].strip().lower()
        if len(short) > 3:
            virus_short_names.add(short)

print(f"  Loaded {len(virus_names)} full virus names")
print(f"  Extracted {len(virus_short_names)} short virus names for matching")

# Create pathogen list (common pathogens in biomedical literature)
COMMON_PATHOGENS = {
    # Bacteria
    'escherichia coli', 'e. coli', 'staphylococcus aureus', 's. aureus',
    'streptococcus pneumoniae', 'salmonella', 'salmonella typhimurium',
    'pseudomonas aeruginosa', 'mycobacterium tuberculosis', 'bacillus',
    'clostridium', 'listeria', 'vibrio', 'vibrio cholerae', 'helicobacter pylori',
    'campylobacter', 'legionella', 'bordetella', 'borrelia', 'treponema',
    'neisseria', 'haemophilus', 'klebsiella', 'enterococcus', 'rickettsia',
    'chlamydia', 'mycoplasma', 'brucella', 'yersinia', 'shigella',

    # Parasites
    'plasmodium', 'plasmodium falciparum', 'plasmodium vivax',
    'trypanosoma', 'trypanosoma cruzi', 'trypanosoma brucei',
    'leishmania', 'toxoplasma', 'toxoplasma gondii',
    'giardia', 'entamoeba', 'cryptosporidium', 'schistosoma',
    'ascaris', 'trichinella', 'strongyloides', 'ancylostoma',
    'echinococcus', 'taenia', 'fasciola', 'opisthorchis',
    'eimeria', 'babesia', 'theileria', 'anaplasma',

    # Fungi
    'candida', 'candida albicans', 'aspergillus', 'aspergillus fumigatus',
    'cryptococcus', 'histoplasma', 'blastomyces', 'coccidioides',
    'pneumocystis', 'fusarium', 'trichophyton', 'microsporum',

    # Common viruses (short names)
    'influenza', 'hiv', 'hepatitis', 'herpes', 'coronavirus', 'sars',
    'ebola', 'dengue', 'zika', 'rabies', 'measles', 'mumps', 'rubella',
    'polio', 'rotavirus', 'norovirus', 'adenovirus', 'papillomavirus',
    'cytomegalovirus', 'cmv', 'epstein-barr', 'ebv', 'varicella', 'vzv',
}

# Combine with virus short names
all_pathogens = COMMON_PATHOGENS | virus_short_names
print(f"  Total pathogen terms: {len(all_pathogens)}")

# Save pathogen list
pathogen_file = analysis_dir / "pathogen_list.txt"
with open(pathogen_file, 'w', encoding='utf-8') as f:
    for p in sorted(all_pathogens):
        f.write(p + '\n')
print(f"  Saved pathogen list to: {pathogen_file}")

# =============================================================================
# STEP 2: Load interaction dictionary
# =============================================================================
print("\n[2] LOADING INTERACTIONS...")

interactions = []
with open(interaction_dict_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if row:
            interactions.append(row[0].strip().lower())

# Separate strong interactions from noisy ones
NOISY_TERMS = {
    'ate', 'eat', 'eated', 'eating', 'eats', 'eat by', 'eated by',
    'affecting', 'affected by', 'regulate', 'regulates', 'regulated by',
    'interacts with', 'interact with', 'depends on', 'contribute to',
    'adjacent to', 'co-occurs with', 'contributing to',
}

strong_interactions = [i for i in interactions if i not in NOISY_TERMS]
print(f"  Total interactions: {len(interactions)}")
print(f"  Strong interactions (excluding noisy): {len(strong_interactions)}")

# =============================================================================
# STEP 3: Define species detection patterns
# =============================================================================
print("\n[3] SETTING UP SPECIES DETECTION...")

# Common species patterns (binomial nomenclature + common names)
SPECIES_PATTERNS = [
    # Binomial names: Genus species
    r'\b[A-Z][a-z]+\s+[a-z]{2,}\b',
    # Abbreviated: G. species
    r'\b[A-Z]\.\s*[a-z]{2,}\b',
]

# Common organism keywords that indicate species context
ORGANISM_KEYWORDS = {
    'bacteria', 'bacterium', 'virus', 'viral', 'fungus', 'fungi', 'fungal',
    'parasite', 'parasitic', 'host', 'pathogen', 'microbe', 'microbial',
    'protozoa', 'protozoan', 'helminth', 'nematode', 'cestode', 'trematode',
    'arthropod', 'insect', 'tick', 'mite', 'flea', 'mosquito', 'fly',
    'mammal', 'bird', 'fish', 'reptile', 'amphibian', 'invertebrate',
    'plant', 'animal', 'organism', 'species', 'strain', 'isolate',
    'human', 'mouse', 'mice', 'rat', 'rabbit', 'pig', 'chicken', 'cattle', 'sheep',
    'dog', 'cat', 'horse', 'primate', 'monkey', 'chimpanzee',
}

def count_species_mentions(sentence):
    """Count likely species mentions in a sentence."""
    count = 0
    s_lower = sentence.lower()

    # Check for binomial names
    for pattern in SPECIES_PATTERNS:
        matches = re.findall(pattern, sentence)
        count += len(matches)

    # Check for pathogens
    for pathogen in all_pathogens:
        if pathogen in s_lower:
            count += 1

    # Check for organism keywords
    for keyword in ORGANISM_KEYWORDS:
        if keyword in s_lower:
            count += 0.5  # Partial credit

    return count

def has_strong_interaction(sentence):
    """Check if sentence has a strong (non-noisy) interaction term."""
    s_lower = sentence.lower()
    for term in strong_interactions:
        if term in s_lower:
            # Special case: "phoresis" but not "electrophoresis"
            if 'phoresis' in term and 'electrophoresis' in s_lower:
                continue
            return True, term
    return False, None

# =============================================================================
# STEP 4: Load and classify sentences
# =============================================================================
print("\n[4] LOADING AND CLASSIFYING SENTENCES...")

positives = []
negatives = []

with open(training_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['label']) == 1:
            positives.append(row['passage'])
        else:
            negatives.append(row['passage'])

print(f"  Original positives: {len(positives)}")
print(f"  Original negatives: {len(negatives)}")

# =============================================================================
# STEP 5: Validate positives (need interaction + species context)
# =============================================================================
print("\n[5] VALIDATING POSITIVES...")

validated_positives = []
weak_positives = []

for sentence in positives:
    has_interaction, term = has_strong_interaction(sentence)
    species_score = count_species_mentions(sentence)

    # A valid positive needs:
    # 1. A strong interaction term
    # 2. At least 2 species-like mentions (score >= 2)
    if has_interaction and species_score >= 2:
        validated_positives.append(sentence)
    elif has_interaction and species_score >= 1:
        # Borderline - keep but flag
        validated_positives.append(sentence)
    else:
        weak_positives.append((sentence, has_interaction, species_score))

print(f"  Validated positives: {len(validated_positives)}")
print(f"  Weak positives (removed): {len(weak_positives)}")

# =============================================================================
# STEP 6: Validate negatives (should NOT have strong interaction + 2 species)
# =============================================================================
print("\n[6] VALIDATING NEGATIVES...")

validated_negatives = []
false_negatives = []

for sentence in negatives:
    has_interaction, term = has_strong_interaction(sentence)
    species_score = count_species_mentions(sentence)

    # A false negative would have:
    # 1. A strong interaction term
    # 2. At least 2 species mentions
    if has_interaction and species_score >= 2:
        false_negatives.append((sentence, term, species_score))
    else:
        validated_negatives.append(sentence)

print(f"  Validated negatives: {len(validated_negatives)}")
print(f"  False negatives (promote to positive): {len(false_negatives)}")

# Show examples of false negatives
print("\n  Examples of FALSE NEGATIVES (should be positive):")
for i, (sent, term, score) in enumerate(false_negatives[:5], 1):
    print(f"    {i}. Term: '{term}', Species score: {score:.1f}")
    print(f"       {sent[:100]}...")

# =============================================================================
# STEP 7: Build final dataset
# =============================================================================
print("\n[7] BUILDING FINAL DATASET...")

# Final positives = validated + promoted false negatives
final_positives = validated_positives + [s for s, _, _ in false_negatives]
final_negatives = validated_negatives

print(f"  Final positives: {len(final_positives)}")
print(f"  Final negatives: {len(final_negatives)}")

# Balance the dataset
min_size = min(len(final_positives), len(final_negatives))
target_size = min(min_size, 10000)

random.shuffle(final_positives)
random.shuffle(final_negatives)

balanced_positives = final_positives[:target_size]
balanced_negatives = final_negatives[:target_size]

print(f"\n  Balanced dataset:")
print(f"    Positives: {len(balanced_positives)}")
print(f"    Negatives: {len(balanced_negatives)}")
print(f"    Total: {len(balanced_positives) + len(balanced_negatives)}")

# =============================================================================
# STEP 8: Quality check samples
# =============================================================================
print("\n[8] QUALITY CHECK...")

print("\n  Sample POSITIVES:")
for i, sent in enumerate(random.sample(balanced_positives, min(5, len(balanced_positives))), 1):
    has_int, term = has_strong_interaction(sent)
    score = count_species_mentions(sent)
    print(f"    {i}. [Int: {term}, Species: {score:.1f}] {sent[:100]}...")

print("\n  Sample NEGATIVES:")
for i, sent in enumerate(random.sample(balanced_negatives, min(5, len(balanced_negatives))), 1):
    has_int, term = has_strong_interaction(sent)
    score = count_species_mentions(sent)
    print(f"    {i}. [Int: {term}, Species: {score:.1f}] {sent[:100]}...")

# =============================================================================
# STEP 9: Save the dataset
# =============================================================================
print("\n[9] SAVING DATASET...")

output_file = output_dir / "training_data_v3_validated.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['passage', 'label'])
    for sent in balanced_positives:
        writer.writerow([sent, 1])
    for sent in balanced_negatives:
        writer.writerow([sent, 0])

print(f"  Saved: {output_file}")

# Save weak positives for review
weak_file = analysis_dir / "weak_positives_v3.csv"
with open(weak_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['sentence', 'has_interaction', 'species_score'])
    for sent, has_int, score in weak_positives:
        writer.writerow([sent, has_int, f'{score:.1f}'])
print(f"  Saved weak positives: {weak_file}")

# Save promoted false negatives
fn_file = analysis_dir / "promoted_false_negatives_v3.csv"
with open(fn_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['sentence', 'interaction_term', 'species_score'])
    for sent, term, score in false_negatives:
        writer.writerow([sent, term, f'{score:.1f}'])
print(f"  Saved promoted false negatives: {fn_file}")

# =============================================================================
# STEP 10: Summary
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
DATASET CHANGES:
  Original positives: {len(positives)}
    - Removed weak (no interaction or few species): {len(weak_positives)}
    + Added promoted false negatives: {len(false_negatives)}
    = Validated positives: {len(final_positives)}

  Original negatives: {len(negatives)}
    - Promoted to positives: {len(false_negatives)}
    = Validated negatives: {len(final_negatives)}

FINAL BALANCED DATASET:
  Positives: {len(balanced_positives)}
  Negatives: {len(balanced_negatives)}
  Total: {len(balanced_positives) + len(balanced_negatives)}

VALIDATION CRITERIA:
  Positive = Strong interaction term + Species mentions (score >= 1)
  Negative = No strong interaction OR insufficient species context

PATHOGEN LIST:
  Total pathogens: {len(all_pathogens)}
  Saved to: {pathogen_file}
""")

print("="*80)

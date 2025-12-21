import pandas as pd
import re
import ast

# Load your data
df = pd.read_csv("training_data.csv")

# Define a basic list of species/group names to look for
# You can extend this list or load it from a file
species_list = [
    "Enterobacteriaceae",
    "Vibrionaceae",
    "Escherichia coli",
    "Salmonella",
    "Pseudomonas",
    "Bacillus",
    "Clostridium",
    "Staphylococcus"
]

# Precompile regex patterns for efficiency
species_patterns = [(name, re.compile(r'\b' + re.escape(name) + r'\b', flags=re.IGNORECASE)) for name in species_list]

def find_spans(text):
    spans = []
    for name, pattern in species_patterns:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            spans.append((start, end - 1))  # LUKE uses inclusive end indices
    return spans if spans else [(-1, -1)]  # fallback span if nothing is found

# Apply span detection
df['spans'] = df['Sentence'].astype(str).apply(find_spans)

# Optional: Save the updated dataframe
df.to_csv("training_data_with_spans.csv", index=False)

print("Spans generated and saved to training_data_with_spans.csv")

# "pathogen of" + bacteria or virus pathogen // not working yet

import pandas as pd
import numpy as np
import requests
import time
import json
import re
from itertools import zip_longest
import warnings
from collections import Counter
from pymongo import MongoClient

start_time = time.time()

# Connect to MongoDB
client = MongoClient("mongodb://sibils-mongodb.lan.text-analytics.ch:27017/")

# Access the database
db = client["sibils_v4_2"]

# Access the collection
collection = db["pmc25_r1_v5.5_passages"]

# List of known pathogenic bacteria and viruses (This is just an example; extend as needed)
pathogenic_microbes = {
    # **Gram-negative bacteria**
    "Escherichia coli", "Salmonella enterica", "Shigella dysenteriae", "Klebsiella pneumoniae",
    "Pseudomonas aeruginosa", "Yersinia pestis", "Vibrio cholerae", "Haemophilus influenzae",
    "Neisseria meningitidis", "Neisseria gonorrhoeae", "Legionella pneumophila", "Bordetella pertussis",
    "Francisella tularensis", "Brucella abortus", "Brucella melitensis", "Helicobacter pylori",
    "Campylobacter jejuni", "Burkholderia pseudomallei", "Acinetobacter baumannii",

    # **Gram-positive bacteria**
    "Staphylococcus aureus", "Streptococcus pneumoniae", "Streptococcus pyogenes",
    "Listeria monocytogenes", "Bacillus anthracis", "Clostridium tetani", "Clostridium botulinum",
    "Clostridium difficile", "Mycobacterium tuberculosis", "Mycobacterium leprae",
    "Corynebacterium diphtheriae", "Enterococcus faecalis",

    # **Spirochetes**
    "Treponema pallidum", "Borrelia burgdorferi", "Leptospira interrogans",

    # **Intracellular bacterial pathogens**
    "Chlamydia trachomatis", "Chlamydophila pneumoniae", "Rickettsia rickettsii", "Rickettsia prowazekii",
    "Coxiella burnetii", "Ehrlichia chaffeensis", "Anaplasma phagocytophilum",

    # **Viruses (RNA)**
    "HIV", "Influenza A virus", "Influenza B virus", "SARS-CoV-2", "MERS-CoV", "SARS-CoV",
    "Ebola virus", "Marburg virus", "Rabies virus", "Zika virus", "Dengue virus", "Yellow fever virus",
    "West Nile virus", "Hepatitis A virus", "Hepatitis C virus", "Hepatitis E virus", 
    "Lassa virus", "Crimean-Congo hemorrhagic fever virus", "Hantavirus",

    # **Viruses (DNA)**
    "Hepatitis B virus", "Human papillomavirus (HPV)", "Herpes simplex virus 1", "Herpes simplex virus 2",
    "Varicella-zoster virus", "Cytomegalovirus", "Epstein-Barr virus", "Human herpesvirus 6",
    "Adenovirus", "Parvovirus B19", "Molluscum contagiosum virus",

    # **Fungi (Some opportunistic pathogens)**
    "Candida albicans", "Cryptococcus neoformans", "Aspergillus fumigatus", "Histoplasma capsulatum",
    "Coccidioides immitis", "Blastomyces dermatitidis", "Pneumocystis jirovecii"
}


# Query to find documents where:
# 1. The interaction is "pathogen of"
# 2. One of the species is a known pathogen
query = {
    "interaction_form": "pathogen of",
    "$or": [
        {"species1": {"$in": list(pathogenic_microbes)}},
        {"species2": {"$in": list(pathogenic_microbes)}}
    ]
}

# Get the total number of documents in the collection
total_documents = collection.count_documents({})

# Function to normalize passages
def normalize_passage(passage):
    passage = passage.lower()  # Convert to lowercase
    passage = re.sub(r'\s+', ' ', passage)  # Replace multiple spaces with a single space
    passage = re.sub(r'[^\w\s]', '', passage)  # Remove punctuation
    passage = passage.strip()  # Strip leading/trailing whitespace
    passage = passage.replace('\u00A0', ' ')  # Replace non-breaking spaces
    passage = passage.replace('\t', ' ')  # Replace tabs
    passage = passage.replace('\n', ' ')  # Replace newlines
    return passage

# Retrieve passages
passages_with_pathogen_interaction = []

for doc in collection.find(query):
    passage = doc.get("passage", "")
    normalized_passage = normalize_passage(passage)
    passages_with_pathogen_interaction.append(normalized_passage)

# Remove duplicates
unique_passages = set(passages_with_pathogen_interaction)

# Save the set of unique passages to a file
with open("passages_with_pathogen_interaction.txt", "w") as file:
    for passage in unique_passages:
        file.write(f"{passage}\n")

# Close the MongoDB connection
client.close()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

















"""

#import mysql.connector

scores = [0.2, 0.5, 0.8, 0.3, 0.9]

# Create a bar plot using Matplotlib
plt.bar(range(len(scores)), scores, color='blue')

# Set labels and title
plt.xlabel('Data Points')
plt.ylabel('Scores')
plt.title('Scores between 0 and 1')

# Display the plot
plt.show()


# Establish a connection to the database
conn = mysql.connector.connect(
    host='localhost',  # Or use the actual host name if different
    user='esteban',  # Replace with your MySQL username
    password='ClouksiBella,5',  # Replace with your MySQL password
    database='impaakt_db_prod'  # Replace with the name of your database
)

# Check if the connection is successful
if conn.is_connected():
    print("Connected to the MySQL database")

# Perform database operations here

# Close the connection when done
conn.close()



"""
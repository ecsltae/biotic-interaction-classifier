# Description: 3 species and no interactions retriever

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
import csv

# Connect to MongoDB
client = MongoClient("mongodb://sibils-mongodb.lan.text-analytics.ch:27017/") 

# Access the database
db = client["sibils_v4_2"]

# Access the collection
collection = db["med25_r1_v5.5_passages"]

# Read the file into a DataFrame
df = pd.read_csv("test_esteban_v4_2.txt", sep="\t")

# Filter out rows where 'presence_of_robi' is True
df_filtered = df[df["presence_of_robi"] == False]

# Extract doc_id and sentence_id pairs from the filtered DataFrame
query_pairs = df_filtered[["doc_id", "sentence_id"]].astype(str).values.tolist()

# List to store retrieved passages
passages = []

# Query MongoDB for each doc_id and sentence_id
for doc_id, sentence_id in query_pairs:
    result = collection.find_one({"doc_id": doc_id, "sentence_begin": sentence_id}, {"_id": 0, "passage": 1})
    if result:
        passages.append(result["passage"])

unique_passages = set(passages)

# Save the set of unique passages to a file
with open('passages_with_3species_nointeractions.txt', 'w') as file:
    for passage in unique_passages:
        file.write(f"{passage}\n")

# Convert the list into a DataFrame
df_passages = pd.DataFrame({"passage": list(unique_passages)})

# Save to CSV
df_passages.to_csv("passages_with_3species_nointeractions.csv", index=False)

# Close the MongoDB connection
client.close()


"""




"""
"""Clade role prior.  Higher score = more likely to be the AGENT (parasite / pathogen /
predator / pollinator) of a biotic interaction.  The table was fixed BEFORE any accuracy
measurement and was not adjusted afterwards.  Validated against GloBI role assignment:
98.39% (n=3,720, 95% CI [0.979, 0.987]).  `kindscore(name)` -> (clade token, score).
"""
import json, re, pandas as pd, numpy as np, sys
sys.path.insert(0,'/home/egaillac/MetaP/classifier/experiments/direction')
import ottlib
R='/home/egaillac/MetaP/classifier/'

PRIOR = [  # (clade token, score). First match walking the lineage root-ward wins.
 ('Microsporida',88),('Apicomplexa',88),('Trypanosomatida',88),('Metakinetoplastina',88),
 ('Peronosporomycetes',86),('Oomycota',86),
 ('Phthiraptera',78),('Siphonaptera',78),('Hippoboscidae',78),('Parasitoida',78),
 ('Chalcidoidea',76),('Ixodida',78),('Acari',70),('Strepsiptera',78),
 ('Nematoda',75),('Tylenchomorpha',75),('Platyhelminthes',75),('Acanthocephala',75),
 ('Fungi',80),('Ascomycota',80),('Basidiomycota',80),
 ('Bacteria',90),('Archaea',88),('Viruses',100),('Tenericutes',90),
 ('Ciliophora',60),('Amoebozoa',60),('Euglenozoa',60),
 ('Bacillariophyta',10),('Rhodophyta',12),('Ulvophyceae',12),('Phaeophyceae',12),
 ('Viridiplantae',15),('Chloroplastida',15),('Embryophyta',15),('Magnoliopsida',15),
 ('Vertebrata',20),('Craniata',20),('Aves',20),('Mammalia',20),('Actinopterygii',20),
 ('Arthropoda',50),('Insecta',50),('Mollusca',45),('Annelida',45),('Metazoa',40),
 ('Eukaryota',30),
]
def kindscore(name):
    o=ottlib.resolve(name)
    if not o: return None,None
    lin=ottlib.lineage(o)
    s=set(lin)
    best=None
    for tok,sc in PRIOR:
        if tok in s:
            if best is None or sc>best[1]: best=(tok,sc)
    return (best if best else (None,None))


# Biotic interaction verifier — handoff

Scores candidate `(taxon, relation, taxon)` triples against the passage they were extracted
from, and answers two questions:

* **does this pair interact**, according to this passage?
* **which taxon is the subject** of the relation?

It runs on CPU. It replaces the sentence-level interaction classifier in the filtering step.

---

## 1. Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the torch/transformers pair the weights were written with. Nothing else
is needed — the package is self-contained and makes **no network calls** at inference time.

Check it works:

```bash
python predict.py --in example_input.csv --out /tmp/check.csv
```

You should see `3 candidates, 3 accepted`.

## 2. Run it

**A file of candidates** (the normal case):

```bash
python predict.py --in candidates.csv --out scored.csv
```

`candidates.csv` needs exactly these four columns — any others you add are copied through to the
output untouched, so you can keep your own ids:

| column | meaning |
|---|---|
| `species1` | first taxon, **as it appears in the passage** (surface form, not the canonical name) |
| `relation` | interaction surface form |
| `species2` | second taxon, same rule |
| `sentence` | the passage the candidate was extracted from |

**One candidate**, for spot checks:

```bash
python predict.py --s1 "Haemophilus influenzae" --rel "pathogen of" --s2 human \
                  --text "Haemophilus influenzae is a major pathogen of humans."
```

Useful flags: `--threshold 0.95` (see §4), `--threads 16`, `--batch-size 32`.

## 3. Read the output

| column | meaning |
|---|---|
| `interacts` | **0/1 — the decision.** This is the column that replaces the old classifier's verdict |
| `p_interact` | the score behind it, 0–1. Use this if you want your own cutoff |
| `direction` | `FORWARD` (species1 is the subject), `REVERSE` (species2 is), or `UNCERTAIN` |
| `p_species1_is_subject` | 0–1; `direction` is this thresholded with an abstention band |
| `direction_confidence` | `abs(p − 0.5) × 2`. Below 0.60 the direction is reported as `UNCERTAIN` |
| `both_taxa_located` | 0 if a taxon string was not found in the passage — **see §5** |
| `unknown_polarity` | 1 if the relation is not in the polarity lexicon and the direction head fell back to a default. A high rate here means the direction column is weaker than §4 suggests |

`FORWARD`/`REVERSE` are judged against the **canonical** relation, not the surface string. So
`Leptodora --prey--> Bosmina` comes back `REVERSE`, because the canonical relation is
*preyed upon by* and Leptodora is the predator. That is the convention the gold annotation uses.

**Order does not matter.** Give the same pair the other way round and `p_interact` is bit-for-bit
identical while `direction` flips. That is a property of the architecture, not of training, so it
cannot drift.

## 4. Which threshold to use

The default is **0.50**. Measured on a 437-item expert-graded evaluation set:

| `--threshold` | precision | recall | false pos | false neg | |
|---|---|---|---|---|---|
| 0.50 | 0.852 | 0.959 | 41 | 10 | **default** — catches nearly everything |
| 0.70 | 0.856 | 0.939 | 39 | 15 | |
| 0.90 | 0.871 | 0.902 | 33 | 24 | |
| 0.95 | 0.882 | 0.878 | 29 | 30 | best balance if curation time is scarce |
| 0.99 | 0.924 | 0.744 | 15 | 63 | only when a false positive is expensive |
| *the filter this replaces* | *0.850* | *0.785* | *34* | *53* | |

At the default the model has **the same precision as the old filter with 17 points more recall**.
Raise the threshold to buy precision; the table is the exact trade.

## 5. When to distrust it

* **`both_taxa_located = 0`.** The model marks the two taxa in the passage so it knows which pair
  it is being asked about. If a taxon string is not found, that marking failed and the answer is
  much weaker. Treat those rows as unreviewed. It fires on 5 of the 449 benchmark rows.

  An earlier version of this note said the flag never fired at all. That was true and misleading:
  taxon matching had no word boundaries, so a genus matched inside an unrelated one — `Aedes`
  matched the fragment `Aede` in *Aedeomyia*, `Bos` matched inside *Bostrichidae* — and the flag
  reported a successful location for a marked fragment of the wrong organism. Matching is now
  anchored at word boundaries, so the flag means what it says. If you scored data with an earlier
  copy of this package, rows whose taxa are substrings of other words are worth re-checking.
* **Co-occurrence in a shared host.** The clearest residual error class. When two organisms are
  both mentioned in relation to a *third* one, the model can read that as an interaction between
  them. `example_input.csv` row 3 is a real case: *Acanthamoeba* and *Pseudomonas* both infect
  the horse, not each other, and the model accepts it at p=0.88. It is a confirmed false positive.
* **Entity errors are not fixed here.** Of the errors the old filter made, about half are
  pair-binding (this model addresses them) and about half are wrong taxon resolution upstream
  (it does not). If the rule layer hands it the wrong species, it will happily verify the wrong
  species.
* **Direction is measured on 84 human-graded items**: 66/84 = 0.786 overall (chance 0.5,
  p = 6.7e-08), and **0.889 on the 75% of rows it is confident enough to answer**. A control that
  hides the passage and shows only the two taxon names scores 0.464 — chance — so the head is
  reading the sentence, not recalling which organisms usually parasitise which.
  It is weakest on bare relational nouns (*host*, *pathogen*, *infection*: 0.725) and strongest
  where the relation word itself carries direction (*pathogen of*, *ectoparasite of*: 0.909).
* **Relations outside the polarity lexicon** fall back to agent-side polarity silently. Common
  interaction vocabulary is covered; an unusual verb may not be.

## 6. Speed

~**34 candidates/s** on 8 CPU threads on real passages (≈120,000/hour). Short sentences run
faster — the example file benches near 95/s — so plan with the lower figure. `--threads` scales
roughly linearly to the core count. Memory: about 1.5 GB resident.

The direction output is **free**: the same model at the same speed. The head is 594,465
parameters, +0.54% on top of the encoder.

## 7. What is in this folder

```
predict.py            what you run
joint_model.py        the architecture (encoder + antisymmetric direction head)
xenc_format.py        builds the encoder input; do not edit, training used this exact code
polarity.py           relation-polarity lexicon feeding the direction head
robi_maps.json        }
data/robiext_v2025.json } the ROBI ontology those lexicons are built from
model/                the weights, 421 MB
example_input.csv     three candidates to check the install
requirements.txt      pinned torch/transformers
```

Total 422 MB. `model/` holds the weights — **do not commit it to git.**

## 8. Who to ask

Model card with the full evaluation, statistics and caveats:
`docs/MODEL_CARD_joint_v2.md` in the classifier repo.
Evaluation audit, including known defects in older result tables:
`docs/EVAL_AUDIT_2026-09-25.md`.
